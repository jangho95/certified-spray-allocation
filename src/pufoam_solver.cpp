#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  int field_size = 50;
  int waypoint_rows = 25;
  int waypoint_cols = 25;
  int spray_interval = 2;
  int kernel_size = 7;
  double sigma_x = 1.75;
  double sigma_y = 1.75;
  double stamp_mass = -1.0;
  int cmax = 200;
  double target = 200.0;
  std::string field_type = "zero";
                                                                     
                                                                               
                                                                        
  std::string boundary = "reflect";
  double center_height = 100.0;
  int center_size = 16;
  double random_low = 1.0;
  double random_high = 30.0;
  uint32_t seed = 1;
  long long budget = -1;
  long long max_exchanges = 200000;
  double min_improvement = 1e-9;
  std::string out_prefix = "solution";
  std::string write_initial;
                                                                               
                                                                            
  std::string initial_field;
                                                                              
                                                          
  std::string init_allocation;
                                                                         
                                                                            
                                                      
  std::string eval_allocation;
  std::string eval_field_out;
  std::string eval_mode = "pershot";                                
  std::string dump_operator;
  uint32_t eval_seed = 12345;
  bool full_precision = false;
  bool quiet = false;
};

struct Stamp {
  std::vector<int> cell;
  std::vector<double> weight;
  double sum = 0.0;
  double self_dot = 0.0;
};

struct Metrics {
  double mean = 0.0;
  double variance = 0.0;
  double min_value = 0.0;
  double max_value = 0.0;
  long long total_shots = 0;
};

void print_usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n\n"
      << "Options:\n"
      << "  --field zero|random|center       Initial field type (default: zero)\n"
      << "  --target VALUE                   Target mean thickness (default: 200)\n"
      << "  --budget INT                     Override total shot budget\n"
      << "  --field-size INT                 Field side length (default: 50)\n"
      << "  --waypoint-rows INT              Waypoint grid rows (default: 25)\n"
      << "  --waypoint-cols INT              Waypoint grid columns (default: 25)\n"
      << "  --spray-interval INT             Waypoint spacing in cells (default: 2)\n"
      << "  --kernel-size INT                Gaussian kernel side length (default: 7)\n"
      << "  --cmax INT                       Maximum shots per waypoint (default: 200)\n"
      << "  --seed INT                       RNG seed for random field (default: 1)\n"
      << "  --center-height VALUE            Center-square initial height (default: 100)\n"
      << "  --center-size INT                Center-square side length (default: 16)\n"
      << "  --random-low VALUE               Random field lower bound (default: 1)\n"
      << "  --random-high VALUE              Random field upper bound (default: 30)\n"
      << "  --sigma VALUE                    Symmetric Gaussian sigma (default: 1.75)\n"
      << "  --sigma-x VALUE                  Gaussian sigma x (default: 1.75)\n"
      << "  --sigma-y VALUE                  Gaussian sigma y (default: 1.75)\n"
      << "  --boundary reflect|truncate      Boundary handling (default: reflect)\n"
      << "  --stamp-mass VALUE               Rescale every folded stamp to this mass\n"
      << "  --max-exchanges INT              Exchange-repair move cap (default: 200000)\n"
      << "  --out-prefix PATH                Output CSV prefix (default: solution)\n"
      << "  --write-initial PATH             Write the generated initial field CSV\n"
      << "  --initial-field PATH             Load the initial field from CSV (overrides --field)\n"
      << "  --init-allocation PATH           Skip greedy; repair from this allocation CSV\n"
      << "  --eval-allocation PATH           Audit: evaluate this allocation CSV and exit\n"
      << "  --eval-field-out PATH            Audit: write the evaluated field CSV\n"
      << "  --eval-mode MODE                 Audit: pershot|scaled|shuffled (default: pershot)\n"
      << "  --eval-seed INT                  Audit: RNG seed for shuffled mode (default: 12345)\n"
      << "  --dump-operator PATH             Audit: write sim(1+e_j)-sim(1) for all j and exit\n"
      << "  --full-precision                 Write floating outputs with 17 significant digits\n"
      << "  --quiet                          Print only final metrics\n"
      << "  --help                           Show this help\n";
}

template <class T>
T parse_value(const std::string& s) {
  std::istringstream iss(s);
  T value{};
  iss >> value;
  if (!iss || !iss.eof()) {
    throw std::runtime_error("Could not parse value: " + s);
  }
  return value;
}

Options parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
      return argv[++i];
    };

    if (a == "--help" || a == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (a == "--field") {
      opt.field_type = need(a);
    } else if (a == "--target") {
      opt.target = parse_value<double>(need(a));
    } else if (a == "--budget") {
      opt.budget = parse_value<long long>(need(a));
    } else if (a == "--field-size") {
      opt.field_size = parse_value<int>(need(a));
    } else if (a == "--waypoint-rows") {
      opt.waypoint_rows = parse_value<int>(need(a));
    } else if (a == "--waypoint-cols") {
      opt.waypoint_cols = parse_value<int>(need(a));
    } else if (a == "--spray-interval") {
      opt.spray_interval = parse_value<int>(need(a));
    } else if (a == "--kernel-size") {
      opt.kernel_size = parse_value<int>(need(a));
    } else if (a == "--cmax") {
      opt.cmax = parse_value<int>(need(a));
    } else if (a == "--seed") {
      opt.seed = parse_value<uint32_t>(need(a));
    } else if (a == "--center-height") {
      opt.center_height = parse_value<double>(need(a));
    } else if (a == "--center-size") {
      opt.center_size = parse_value<int>(need(a));
    } else if (a == "--random-low") {
      opt.random_low = parse_value<double>(need(a));
    } else if (a == "--random-high") {
      opt.random_high = parse_value<double>(need(a));
    } else if (a == "--sigma") {
      opt.sigma_x = opt.sigma_y = parse_value<double>(need(a));
    } else if (a == "--sigma-x") {
      opt.sigma_x = parse_value<double>(need(a));
    } else if (a == "--sigma-y") {
      opt.sigma_y = parse_value<double>(need(a));
    } else if (a == "--boundary") {
      opt.boundary = need(a);
      if (opt.boundary != "reflect" && opt.boundary != "truncate") {
        throw std::runtime_error("--boundary must be reflect or truncate");
      }
    } else if (a == "--stamp-mass") {
      opt.stamp_mass = parse_value<double>(need(a));
    } else if (a == "--max-exchanges") {
      opt.max_exchanges = parse_value<long long>(need(a));
    } else if (a == "--out-prefix") {
      opt.out_prefix = need(a);
    } else if (a == "--write-initial") {
      opt.write_initial = need(a);
    } else if (a == "--initial-field") {
      opt.initial_field = need(a);
    } else if (a == "--init-allocation") {
      opt.init_allocation = need(a);
    } else if (a == "--eval-allocation") {
      opt.eval_allocation = need(a);
    } else if (a == "--eval-field-out") {
      opt.eval_field_out = need(a);
    } else if (a == "--eval-mode") {
      opt.eval_mode = need(a);
      if (opt.eval_mode != "pershot" && opt.eval_mode != "scaled" &&
          opt.eval_mode != "shuffled") {
        throw std::runtime_error("--eval-mode must be pershot, scaled or shuffled");
      }
    } else if (a == "--eval-seed") {
      opt.eval_seed = parse_value<uint32_t>(need(a));
    } else if (a == "--dump-operator") {
      opt.dump_operator = need(a);
    } else if (a == "--full-precision") {
      opt.full_precision = true;
    } else if (a == "--quiet") {
      opt.quiet = true;
    } else {
      throw std::runtime_error("Unknown option: " + a);
    }
  }

  if (opt.field_type != "zero" && opt.field_type != "random" &&
      opt.field_type != "center") {
    throw std::runtime_error("--field must be zero, random, or center");
  }
  if (opt.kernel_size % 2 == 0) {
    throw std::runtime_error("kernel size must be odd");
  }
    if (opt.random_low > opt.random_high) {
    throw std::runtime_error("--random-low must be <= --random-high");
  }
  if (opt.stamp_mass == 0.0) {
    throw std::runtime_error("--stamp-mass must be positive when provided");
  }
  return opt;
}

int reflect_index(int p, int n) {
                                                                   
  while (p < 0 || p >= n) {
    if (p < 0) p = -p - 1;
    if (p >= n) p = 2 * n - p - 1;
  }
  return p;
}

std::vector<std::pair<int, int>> make_waypoints(const Options& opt) {
  if (opt.field_size <= 0 || opt.waypoint_rows <= 0 || opt.waypoint_cols <= 0 ||
      opt.spray_interval <= 0 || opt.kernel_size <= 0 || opt.cmax < 1) {
    throw std::runtime_error("Invalid grid/kernel/box option");
  }
  std::vector<std::pair<int, int>> pts;
  pts.reserve(opt.waypoint_rows * opt.waypoint_cols);
  for (int r = 0; r < opt.waypoint_rows; ++r) {
    for (int c = 0; c < opt.waypoint_cols; ++c) {
      const int rr = r * opt.spray_interval;
      const int cc = c * opt.spray_interval;
      if (rr >= opt.field_size || cc >= opt.field_size) {
        throw std::runtime_error("Waypoint grid exceeds field; reduce rows/cols or interval");
      }
      pts.push_back({rr, cc});
    }
  }
  return pts;
}

std::vector<Stamp> build_stamps(const Options& opt,
                                const std::vector<std::pair<int, int>>& pts) {
  const int F = opt.field_size;
  const int radius = opt.kernel_size / 2;
  std::vector<Stamp> stamps;
  stamps.reserve(pts.size());

  for (const auto& [wr, wc] : pts) {
    std::vector<double> dense(F * F, 0.0);
    for (int dr = -radius; dr <= radius; ++dr) {
      for (int dc = -radius; dc <= radius; ++dc) {
        const double z =
            (static_cast<double>(dr * dr) / (2.0 * opt.sigma_y * opt.sigma_y)) +
            (static_cast<double>(dc * dc) / (2.0 * opt.sigma_x * opt.sigma_x));
        const double w = std::exp(-z);
        int rr = wr + dr;
        int cc = wc + dc;
        if (opt.boundary == "truncate") {
                                                                            
                                                                             
          if (rr < 0 || rr >= F || cc < 0 || cc >= F) continue;
        } else {
          rr = reflect_index(rr, F);
          cc = reflect_index(cc, F);
        }
        dense[rr * F + cc] += w;
      }
    }
    if (opt.stamp_mass > 0.0) {
      const double current = std::accumulate(dense.begin(), dense.end(), 0.0);
      const double scale = opt.stamp_mass / current;
      for (double& v : dense) v *= scale;
    }

    Stamp s;
    for (int idx = 0; idx < F * F; ++idx) {
      if (dense[idx] != 0.0) {
        s.cell.push_back(idx);
        s.weight.push_back(dense[idx]);
        s.sum += dense[idx];
        s.self_dot += dense[idx] * dense[idx];
      }
    }
    stamps.push_back(std::move(s));
  }
  return stamps;
}

std::vector<double> read_initial_field(const Options& opt) {
  const int F = opt.field_size;
  std::ifstream in(opt.initial_field);
  if (!in) throw std::runtime_error("Could not open --initial-field " + opt.initial_field);
  std::vector<double> field;
  field.reserve(F * F);
  std::string line;
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      if (tok.find_first_not_of(" \t\r") == std::string::npos) continue;
      field.push_back(std::strtod(tok.c_str(), nullptr));
    }
  }
  if (static_cast<int>(field.size()) != F * F) {
    throw std::runtime_error("--initial-field has " + std::to_string(field.size()) +
                             " values, expected " + std::to_string(F * F));
  }
  return field;
}

std::vector<double> make_initial_field(const Options& opt) {
  if (!opt.initial_field.empty()) return read_initial_field(opt);
  const int F = opt.field_size;
  std::vector<double> field(F * F, 0.0);
  if (opt.field_type == "zero") {
    return field;
  }
  if (opt.field_type == "random") {
    std::mt19937 rng(opt.seed);
    std::uniform_real_distribution<double> dist(opt.random_low, opt.random_high);
    for (double& v : field) v = dist(rng);
    return field;
  }

  const int side = std::min(opt.center_size, F);
  const int r0 = (F - side) / 2;
  const int c0 = (F - side) / 2;
  for (int r = r0; r < r0 + side; ++r) {
    for (int c = c0; c < c0 + side; ++c) {
      field[r * F + c] = opt.center_height;
    }
  }
  return field;
}

double sum_vector(const std::vector<double>& v) {
  return std::accumulate(v.begin(), v.end(), 0.0);
}

std::vector<std::vector<double>> build_gram(const std::vector<Stamp>& stamps,
                                            int field_cells) {
  const int N = static_cast<int>(stamps.size());
  std::vector<std::vector<double>> dense(N, std::vector<double>(N, 0.0));
  std::vector<std::vector<std::pair<int, double>>> by_cell(field_cells);

  for (int j = 0; j < N; ++j) {
    const auto& s = stamps[j];
    for (size_t k = 0; k < s.cell.size(); ++k) {
      by_cell[s.cell[k]].push_back({j, s.weight[k]});
    }
  }

  for (const auto& entries : by_cell) {
    for (const auto& [i, wi] : entries) {
      for (const auto& [j, wj] : entries) {
        dense[i][j] += wi * wj;
      }
    }
  }
  return dense;
}

void add_stamp_to_field(std::vector<double>& field, const Stamp& s, double scale) {
  for (size_t k = 0; k < s.cell.size(); ++k) {
    field[s.cell[k]] += scale * s.weight[k];
  }
}

Metrics compute_metrics(const std::vector<double>& field,
                        const std::vector<int>& x) {
  const int M = static_cast<int>(field.size());
  Metrics m;
  m.mean = sum_vector(field) / static_cast<double>(M);
  m.min_value = std::numeric_limits<double>::infinity();
  m.max_value = -std::numeric_limits<double>::infinity();
  double ss = 0.0;
  for (double v : field) {
    const double d = v - m.mean;
    ss += d * d;
    m.min_value = std::min(m.min_value, v);
    m.max_value = std::max(m.max_value, v);
  }
  m.variance = ss / static_cast<double>(M);
  m.total_shots = std::accumulate(x.begin(), x.end(), 0LL);
  return m;
}

long long choose_budget(const Options& opt,
                        const std::vector<double>& initial,
                        const std::vector<Stamp>& stamps) {
  const int M = opt.field_size * opt.field_size;
  const int N = static_cast<int>(stamps.size());
  const long long lo = N;
  const long long hi = 1LL * N * opt.cmax;
  if (opt.budget >= 0) {
    if (opt.budget < lo || opt.budget > hi) {
      throw std::runtime_error("--budget is outside feasible [N, N*Cmax]");
    }
    return opt.budget;
  }

  const double init_mean = sum_vector(initial) / static_cast<double>(M);
  const double stamp_mass = stamps.front().sum;
  const double raw = (opt.target - init_mean) * static_cast<double>(M) / stamp_mass;
  long long budget = static_cast<long long>(std::llround(raw));
  budget = std::max(lo, std::min(hi, budget));
  return budget;
}

std::vector<int> greedy_construct(const Options& opt,
                                  long long budget,
                                  const std::vector<Stamp>& stamps,
                                  const std::vector<std::vector<double>>& gram,
                                  std::vector<double>& field,
                                  std::vector<double>& dot_r) {
  const int N = static_cast<int>(stamps.size());
  const int M = opt.field_size * opt.field_size;
  std::vector<int> x(N, 1);
  for (int j = 0; j < N; ++j) add_stamp_to_field(field, stamps[j], 1.0);

  const double fixed_mean = sum_vector(field) / static_cast<double>(M) +
                            (static_cast<double>(budget - N) * stamps.front().sum) /
                                static_cast<double>(M);

  std::vector<double> residual = field;
  for (double& v : residual) v -= fixed_mean;

  dot_r.assign(N, 0.0);
  for (int j = 0; j < N; ++j) {
    const auto& s = stamps[j];
    for (size_t k = 0; k < s.cell.size(); ++k) {
      dot_r[j] += residual[s.cell[k]] * s.weight[k];
    }
  }

  long long remaining = budget - N;
  while (remaining > 0) {
    int best = -1;
    double best_delta = std::numeric_limits<double>::infinity();
    for (int j = 0; j < N; ++j) {
      if (x[j] >= opt.cmax) continue;
      const double delta = 2.0 * dot_r[j] + gram[j][j];
      if (delta < best_delta) {
        best_delta = delta;
        best = j;
      }
    }
    if (best < 0) throw std::runtime_error("No feasible waypoint for remaining shots");

    ++x[best];
    --remaining;
    add_stamp_to_field(field, stamps[best], 1.0);
    for (int k = 0; k < N; ++k) dot_r[k] += gram[best][k];
  }

  return x;
}

long long exchange_repair(const Options& opt,
                          const std::vector<Stamp>& stamps,
                          const std::vector<std::vector<double>>& gram,
                          std::vector<int>& x,
                          std::vector<double>& field,
                          std::vector<double>& dot_r) {
  const int N = static_cast<int>(stamps.size());
  long long moves = 0;

  while (moves < opt.max_exchanges) {
    int best_i = -1;
    int best_j = -1;
    double best_delta = -opt.min_improvement;

    for (int i = 0; i < N; ++i) {
      if (x[i] <= 1) continue;
      const double remove_part = -2.0 * dot_r[i] + gram[i][i];
      for (int j = 0; j < N; ++j) {
        if (i == j || x[j] >= opt.cmax) continue;
        const double delta = remove_part + 2.0 * dot_r[j] + gram[j][j] -
                             2.0 * gram[i][j];
        if (delta < best_delta) {
          best_delta = delta;
          best_i = i;
          best_j = j;
        }
      }
    }

    if (best_i < 0) break;

    --x[best_i];
    ++x[best_j];
    add_stamp_to_field(field, stamps[best_i], -1.0);
    add_stamp_to_field(field, stamps[best_j], 1.0);
    for (int k = 0; k < N; ++k) {
      dot_r[k] += gram[best_j][k] - gram[best_i][k];
    }
    ++moves;
  }
  return moves;
}

void write_allocation_csv(const std::string& path,
                          const std::vector<std::pair<int, int>>& pts,
                          const std::vector<int>& x) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("Could not write " + path);
  out << "waypoint,row,col,shot_count\n";
  for (size_t i = 0; i < pts.size(); ++i) {
    out << i << ',' << pts[i].first << ',' << pts[i].second << ',' << x[i] << '\n';
  }
}

                                                                             
                                                                                
                                                                                 
                                      
void write_field_csv(const std::string& path,
                     const std::vector<double>& field,
                     int F,
                     int precision = 10) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("Could not write " + path);
  out << std::setprecision(precision);
  for (int r = 0; r < F; ++r) {
    for (int c = 0; c < F; ++c) {
      if (c) out << ',';
      out << field[r * F + c];
    }
    out << '\n';
  }
}

                                                                              
                                                                            
                                                                             
                                                                              
                       
                                                                              

std::vector<int> read_allocation_csv(const std::string& path, int n) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Could not read " + path);
  std::vector<int> x(n, 0);
  std::string line;
  bool header = true;
  long long rows = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (header) {                                  
      header = false;
      if (!line.empty() && (std::isalpha(static_cast<unsigned char>(line[0])) != 0)) {
        continue;
      }
    }
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) fields.push_back(tok);
    if (fields.size() < 2) {
      throw std::runtime_error("Malformed allocation row: " + line);
    }
                                                                           
    const int j = parse_value<int>(fields.front());
    const int v = parse_value<int>(fields.back());
    if (j < 0 || j >= n) throw std::runtime_error("Waypoint index out of range");
    x[j] = v;
    ++rows;
  }
  if (rows != n) {
    throw std::runtime_error("Allocation CSV has " + std::to_string(rows) +
                             " rows, expected " + std::to_string(n));
  }
  return x;
}

                                                                                 
                                                                            
                           
std::vector<int> init_from_allocation(const Options& opt, long long budget,
                                      const std::vector<Stamp>& stamps,
                                      std::vector<double>& field,
                                      std::vector<double>& dot_r) {
  const int N = static_cast<int>(stamps.size());
  const int M = opt.field_size * opt.field_size;
  std::vector<int> x = read_allocation_csv(opt.init_allocation, N);
  long long total = 0;
  for (int j = 0; j < N; ++j) {
    if (x[j] < 1 || x[j] > opt.cmax) {
      throw std::runtime_error("--init-allocation violates the box at waypoint " + std::to_string(j));
    }
    total += x[j];
  }
  if (total != budget) {
    throw std::runtime_error("--init-allocation has " + std::to_string(total) +
                             " shots, budget is " + std::to_string(budget));
  }
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < x[j]; ++k) add_stamp_to_field(field, stamps[j], 1.0);
  }
  const double fixed_mean = sum_vector(field) / static_cast<double>(M);
  dot_r.assign(N, 0.0);
  for (int j = 0; j < N; ++j) {
    const auto& s = stamps[j];
    for (size_t k = 0; k < s.cell.size(); ++k) dot_r[j] += (field[s.cell[k]] - fixed_mean) * s.weight[k];
  }
  return x;
}

                                                                            
                                                                             
                                        
std::vector<double> evaluate_allocation(const Options& opt,
                                        const std::vector<Stamp>& stamps,
                                        const std::vector<double>& initial,
                                        const std::vector<int>& x) {
  const int N = static_cast<int>(stamps.size());
  std::vector<double> field = initial;

  if (opt.eval_mode == "scaled") {
    for (int j = 0; j < N; ++j) {
      if (x[j] != 0) add_stamp_to_field(field, stamps[j], static_cast<double>(x[j]));
    }
    return field;
  }

  std::vector<int> order;
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < x[j]; ++k) order.push_back(j);
  }
  if (opt.eval_mode == "shuffled") {
    std::mt19937 rng(opt.eval_seed);
    std::shuffle(order.begin(), order.end(), rng);
  }
  for (const int j : order) add_stamp_to_field(field, stamps[j], 1.0);
  return field;
}

                                                                            
                                                                         
                                                                            
                                                   
void dump_operator_columns(const Options& opt,
                           const std::vector<Stamp>& stamps,
                           const std::vector<double>& initial,
                           const std::string& path) {
  const int N = static_cast<int>(stamps.size());
  const std::vector<int> base_x(N, 1);
  const std::vector<double> base = evaluate_allocation(opt, stamps, initial, base_x);

  std::ofstream out(path);
  if (!out) throw std::runtime_error("Could not write " + path);
  out << std::setprecision(17);
  out << "waypoint,cell,delta\n";
  for (int j = 0; j < N; ++j) {
    std::vector<int> xj = base_x;
    xj[j] += 1;
    const std::vector<double> full = evaluate_allocation(opt, stamps, initial, xj);
    for (size_t i = 0; i < full.size(); ++i) {
      const double d = full[i] - base[i];
      if (d != 0.0) out << j << ',' << i << ',' << d << '\n';
    }
  }
}

                                                                            
                                                                                
                                                      
long peak_rss_kb() {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmHWM:", 0) == 0) return std::strtol(line.c_str() + 6, nullptr, 10);
  }
  return -1;
}

void print_metrics(const std::string& label, const Metrics& m,
                   bool full_precision = false) {
  if (full_precision) {
    std::cout << std::defaultfloat << std::setprecision(17);
  } else {
    std::cout << std::fixed << std::setprecision(6);
  }
  std::cout << label << ": mean=" << m.mean
            << " variance=" << m.variance
            << " min=" << m.min_value
            << " max=" << m.max_value
            << " total_shots=" << m.total_shots << '\n';
}

}              

int main(int argc, char** argv) {
  try {
    const auto t0 = std::chrono::steady_clock::now();
    const Options opt = parse_args(argc, argv);
    const int M = opt.field_size * opt.field_size;

    const auto pts = make_waypoints(opt);
    const auto stamps = build_stamps(opt, pts);
    const auto initial = make_initial_field(opt);
    if (!opt.write_initial.empty()) {
                                                                                  
                                                                              
      write_field_csv(opt.write_initial, initial, opt.field_size, 17);
    }

                                               
    if (!opt.dump_operator.empty()) {
      dump_operator_columns(opt, stamps, initial, opt.dump_operator);
      std::cout << "wrote operator columns to " << opt.dump_operator << '\n';
      return 0;
    }
    if (!opt.eval_allocation.empty()) {
      const std::vector<int> x_eval =
          read_allocation_csv(opt.eval_allocation, static_cast<int>(stamps.size()));
      const std::vector<double> field_eval =
          evaluate_allocation(opt, stamps, initial, x_eval);
      const Metrics m = compute_metrics(field_eval, x_eval);
      if (!opt.eval_field_out.empty()) {
        write_field_csv(opt.eval_field_out, field_eval, opt.field_size, 17);
      }
      std::cout << std::setprecision(17)
                << "eval_mode=" << opt.eval_mode
                << " mean=" << m.mean
                << " variance=" << m.variance
                << " total_shots=" << m.total_shots << '\n';
      return 0;
    }

    const double stamp_mass = stamps.front().sum;
    const double init_mean = sum_vector(initial) / static_cast<double>(M);
    const long long budget = choose_budget(opt, initial, stamps);
    const double predicted_mean =
        init_mean + static_cast<double>(budget) * stamp_mass / static_cast<double>(M);

    if (!opt.quiet) {
      std::cout << "PU foam static stop-and-spray solver\n"
                << "field=" << opt.field_type << " F=" << opt.field_size << "x"
                << opt.field_size << " waypoints=" << pts.size()
                << " kernel=" << opt.kernel_size << "x" << opt.kernel_size
                << " sigma=(" << opt.sigma_x << "," << opt.sigma_y << ")\n"
                << "kernel_mass=" << stamp_mass << " init_mean=" << init_mean
                << " target=" << opt.target << " budget=" << budget
                << " predicted_mean=" << predicted_mean << '\n';
    }

    const auto t_gram0 = std::chrono::steady_clock::now();
    const auto gram = build_gram(stamps, M);
    const auto t_gram1 = std::chrono::steady_clock::now();

    std::vector<double> field = initial;
    std::vector<double> dot_r;
    std::vector<int> x = opt.init_allocation.empty()
                             ? greedy_construct(opt, budget, stamps, gram, field, dot_r)
                             : init_from_allocation(opt, budget, stamps, field, dot_r);
    const Metrics greedy_metrics = compute_metrics(field, x);

    const auto t_repair0 = std::chrono::steady_clock::now();
    const long long moves = exchange_repair(opt, stamps, gram, x, field, dot_r);
    const auto t_repair1 = std::chrono::steady_clock::now();
    const Metrics final_metrics = compute_metrics(field, x);

    const std::string alloc_path = opt.out_prefix + "_allocation.csv";
    const std::string field_path = opt.out_prefix + "_field.csv";
    write_allocation_csv(alloc_path, pts, x);
    write_field_csv(field_path, field, opt.field_size,
                    opt.full_precision ? 17 : 10);

    const auto t1 = std::chrono::steady_clock::now();
    const auto ms_total =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    const auto ms_gram =
        std::chrono::duration_cast<std::chrono::milliseconds>(t_gram1 - t_gram0)
            .count();
    const auto ms_repair =
        std::chrono::duration_cast<std::chrono::milliseconds>(t_repair1 - t_repair0)
            .count();

    if (!opt.quiet) print_metrics("after_greedy", greedy_metrics, opt.full_precision);
    print_metrics("final", final_metrics, opt.full_precision);
    std::cout << "exchange_moves=" << moves << " gram_ms=" << ms_gram
              << " repair_ms=" << ms_repair << " total_ms=" << ms_total
              << " peak_rss_kb=" << peak_rss_kb() << '\n'
              << "wrote " << alloc_path << " and " << field_path << '\n';
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}

#include <iostream>
#include <C:\Program Files\Eigen3\Eigen\Dense>
#include <memory>

using namespace Eigen;

class CostBase {
public:
    virtual double evaluate(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const = 0;
    virtual Eigen::VectorXd gradient_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const = 0;
    virtual Eigen::VectorXd gradient_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const = 0;
    virtual Eigen::MatrixXd hessian_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const = 0;
    virtual Eigen::MatrixXd hessian_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const = 0;
};

class ProximityCost : public CostBase {
public:
    double d_threshold;
    int idx1, idx2;
    double weight;

    ProximityCost(double d_threshold = 0.5, int idx1 = 0, int idx2 = 0, double weight = 1.0)
        : d_threshold(d_threshold), idx1(idx1), idx2(idx2), weight(weight) {}

    double evaluate(const VectorXd& x, const VectorXd& u) const {
        double dist = std::sqrt(std::pow(x[4 * idx1] - x[4 * idx2], 2) + std::pow(x[4 * idx1 + 1] - x[4 * idx2 + 1], 2));
        return (dist > d_threshold) ? 0.0 : weight * std::pow((d_threshold - dist),2);
    }

    VectorXd gradient_x(const VectorXd& x, const VectorXd& u) const {
        double dist = std::sqrt(std::pow(x[4 * idx1] - x[4 * idx2], 2) + std::pow(x[4 * idx1 + 1] - x[4 * idx2 + 1], 2));
        if (dist > d_threshold) {
            return VectorXd::Zero(x.size());
        }

        // double denom = -weight / (2 * std::sqrt(std::pow(x[4 * idx1] - x[4 * idx2], 2) + std::pow(x[4 * idx1 + 1] - x[4 * idx2 + 1], 2)) + 1e-6);
        double denom = weight * 2 * (d_threshold - dist);
        VectorXd grad_x = VectorXd::Zero(x.size());
        grad_x[4 * idx1] = 2 * (x[4 * idx1] - x[4 * idx2]) * denom;
        grad_x[4 * idx1 + 1] = 2 * (x[4 * idx1 + 1] - x[4 * idx2 + 1]) * denom;
        grad_x[4 * idx2] = -2 * (x[4 * idx1] - x[4 * idx2]) * denom;
        grad_x[4 * idx2 + 1] = -2 * (x[4 * idx1 + 1] - x[4 * idx2 + 1]) * denom;

        return grad_x;
    }

    VectorXd gradient_u(const VectorXd& x, const VectorXd& u) const {
        return VectorXd::Zero(u.size());
    }

    Eigen::MatrixXd hessian_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        double dist = std::sqrt(std::pow(x[4 * idx1] - x[4 * idx2], 2) + std::pow(x[4 * idx1 + 1] - x[4 * idx2 + 1], 2));
        if (dist > d_threshold) {
            return Eigen::MatrixXd::Zero(x.size(), x.size());
        }
        MatrixXd hessian_x_matrix = MatrixXd::Zero(x.size(), x.size());
        double denom = weight * 2 * (d_threshold - dist);
        for (int i = 0; i < 2; i++) {
            hessian_x_matrix(4 * idx1 + i, 4 * idx1 + i) = 2 * denom;
            hessian_x_matrix(4 * idx1 + i, 4 * idx2 + i) = -2 * denom;
            hessian_x_matrix(4 * idx2 + i, 4 * idx1 + i) = -2 * denom;
            hessian_x_matrix(4 * idx2 + i, 4 * idx2 + i) = 2 * denom;
        }
        return hessian_x_matrix;
    }

    Eigen::MatrixXd hessian_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        return Eigen::MatrixXd::Zero(u.size(), u.size());
    }
};

class ReferenceCost : public CostBase {
public:
    int idx;
    Eigen::VectorXd x_ref;
    double weight;

    ReferenceCost(int idx = 0, const VectorXd& x_ref = VectorXd::Zero(8), double weight = 1.0)
        : idx(idx), x_ref(x_ref), weight(weight) {}

    double evaluate(const VectorXd& x, const VectorXd& u) const {
        double dist = std::pow(std::sqrt(std::pow(x[4 * idx] - x_ref[4 * idx], 2) +
                                std::pow(x[4 * idx + 1] - x_ref[4 * idx + 1], 2) +
                                std::pow(x[4 * idx + 2] - x_ref[4 * idx + 2], 2) +
                                std::pow(x[4 * idx + 3] - x_ref[4 * idx + 3], 2)),2);
        return dist * weight;
    }

    VectorXd gradient_x(const VectorXd& x, const VectorXd& u) const {
        double denom = weight / (2 * std::sqrt(std::pow(x[4 * idx] - x_ref[4 * idx], 2) +
                                              std::pow(x[4 * idx + 1] - x_ref[4 * idx + 1], 2) +
                                              std::pow(x[4 * idx + 2] - x_ref[4 * idx + 2], 2) +
                                              std::pow(x[4 * idx + 3] - x_ref[4 * idx + 3], 2)));

        VectorXd grad_x = VectorXd::Zero(x.size());
        denom = weight;
        grad_x[4 * idx] = 2 * (x[4 * idx] - x_ref[4 * idx]) * denom;
        grad_x[4 * idx + 1] = 2 * (x[4 * idx + 1] - x_ref[4 * idx + 1]) * denom;
        grad_x[4 * idx + 2] = 2 * (x[4 * idx + 2] - x_ref[4 * idx + 2]) * denom;
        grad_x[4 * idx + 3] = 2 * (x[4 * idx + 3] - x_ref[4 * idx + 3]) * denom;

        return grad_x;
    }

    VectorXd gradient_u(const VectorXd& x, const VectorXd& u) const {
        return VectorXd::Zero(u.size());
    }

    Eigen::MatrixXd hessian_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        MatrixXd hessian_x_matrix = MatrixXd::Zero(x.size(), x.size());
        for (int i = 0; i < 4; i++) {
            hessian_x_matrix(4 * idx + i, 4 * idx + i) = 2 * weight;
        }
        return hessian_x_matrix;
    }

        Eigen::MatrixXd hessian_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
            return Eigen::MatrixXd::Zero(u.size(), u.size());
        }
    };

    
class WallCost : public CostBase {
    public:
        int idx;
        double weight;

        WallCost(int idx, double weight = 1.0) : idx(idx), weight(weight) {}

        double evaluate(const VectorXd& x, const VectorXd& u) const {
            double x_robot = x[4 * idx];
            double y_robot = x[4 * idx + 1];

            double side_length = 7.0;
            double x_center = 0.0;
            double y_center = 0.0;

            double dx = std::max(0.0, std::abs(x_robot - x_center) - 0.5 * side_length);
            double dy = std::max(0.0, std::abs(y_robot - y_center) - 0.5 * side_length);

            double dist_penalty = std::pow(std::sqrt(std::pow(dx, 2) + std::pow(dy, 2)), 2);

            return weight * dist_penalty;
        }

        VectorXd gradient_x(const VectorXd& x, const VectorXd& u) const {
            double x_robot = x[4 * idx];
            double y_robot = x[4 * idx + 1];

            double side_length = 7.0;
            double x_center = 0.0;
            double y_center = 0.0;

            double dx = std::max(0.0, std::abs(x_robot - x_center) - 0.5 * side_length);
            double dy = std::max(0.0, std::abs(y_robot - y_center) - 0.5 * side_length);

            double dist_penalty = std::sqrt(std::pow(dx, 2) + std::pow(dy, 2));
            Eigen::VectorXd grad_x = VectorXd::Zero(x.size());

            if (dx > 0) {
                if (x_robot > x_center) {
                    grad_x[4 * idx] = weight * dx * 2;
                } else {
                    grad_x[4 * idx] = -weight * dx * 2;
                }
            }

            if (dy > 0) {
                if (y_robot > y_center) {
                    grad_x[4 * idx + 1] = weight * dy * 2;
                } else {
                    grad_x[4 * idx + 1] = -weight * dy * 2;
                }
            }

            return grad_x;
        }

        VectorXd gradient_u(const VectorXd& x, const VectorXd& u) const {
            return VectorXd::Zero(u.size());
        }

        Eigen::MatrixXd hessian_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
            // calculate the hessian 
            MatrixXd hessian_x_matrix = MatrixXd::Zero(x.size(), x.size());
            double x_robot = x[4 * idx];
            double y_robot = x[4 * idx + 1];

            double side_length = 7.0;
            double x_center = 0.0;
            double y_center = 0.0;

            double dx = std::max(0.0, std::abs(x_robot - x_center) - 0.5 * side_length);
            double dy = std::max(0.0, std::abs(y_robot - y_center) - 0.5 * side_length);

            if (dx > 0) {
                hessian_x_matrix(4 * idx, 4 * idx) = 2 * weight;
            }

            if (dy > 0) {
                hessian_x_matrix(4 * idx + 1, 4 * idx + 1) = 2 * weight;
            }

            return hessian_x_matrix;
          
        }

        Eigen::MatrixXd hessian_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
            return Eigen::MatrixXd::Zero(u.size(), u.size());
        }
    }; 

class InputCost : public CostBase {
public:
    double weight_1;
    double weight_2;

    InputCost(double weight_1 = 1.0, double weight_2 = 1.0) : weight_1(weight_1), weight_2(weight_2) {}

    double evaluate(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        return weight_1 * u[0] * u[0] + weight_2 * u[1] * u[1]; // use weight_1 for the first input and weight_2 for the second
    }

    VectorXd gradient_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        return VectorXd::Zero(x.size());
    }

    VectorXd gradient_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        VectorXd grad_u(u.size());
        grad_u[0] = 2 * weight_1 * u[0]; // use weight_1 for the first input
        grad_u[1] = 2 * weight_2 * u[1]; // use weight_2 for the second input
        return grad_u;
    }

    Eigen::MatrixXd hessian_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        return Eigen::MatrixXd::Zero(x.size(), x.size());
    }

    Eigen::MatrixXd hessian_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        Eigen::MatrixXd hessian_u_matrix = Eigen::MatrixXd::Zero(u.size(), u.size());
        hessian_u_matrix(0, 0) = 2 * weight_1; // use weight_1 for the first input
        hessian_u_matrix(1, 1) = 2 * weight_2; // use weight_2 for the second input
        return hessian_u_matrix;
    }
};

class TrialCost : public CostBase {
public:
    double d_threshold;

    TrialCost(double d_threshold = 0.5) : d_threshold(d_threshold) {}

    double evaluate(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        double dist = std::pow(x[0], 2) + std::pow(x[1], 2) + std::pow(x[2], 2) + std::pow(x[3], 2) +
                      std::pow(x[4], 2) + std::pow(x[5], 2) + std::pow(x[6], 2) + std::pow(x[7], 2);
        return dist;
    }

    VectorXd gradient_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        return VectorXd::Zero(x.size());
    }

    VectorXd gradient_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        return VectorXd::Zero(u.size());
    }

    Eigen::MatrixXd hessian_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        return Eigen::MatrixXd::Zero(x.size(), x.size());
    }

    Eigen::MatrixXd hessian_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        return Eigen::MatrixXd::Zero(u.size(), u.size());
    }
};

class OverallCost {
public:
    std::vector<std::shared_ptr<CostBase>> subsystem_cost_functions;

    OverallCost(const std::vector<std::shared_ptr<CostBase>>& subsystem_cost_functions) : subsystem_cost_functions(subsystem_cost_functions) {
        // Initialize OverallCost with subsystem_cost_functions
        // You can iterate over subsystem_cost_functions and perform necessary operations
        std::cout << "OverallCost constructor called.\n";
    }

    double evaluate(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        double total_cost = 0.0;
        for (const auto& subsystem_cost : subsystem_cost_functions) {
            total_cost += subsystem_cost->evaluate(x, u);
        }
        return total_cost;
    }

    Eigen::VectorXd evaluate_grad(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        Eigen::VectorXd total_cost_grad = Eigen::VectorXd::Zero(x.size());
        for (const auto& subsystem_cost : subsystem_cost_functions) {
            total_cost_grad += subsystem_cost->gradient_x(x, u);
        }
        return total_cost_grad;
    }

    Eigen::VectorXd gradient_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        Eigen::VectorXd grad_x = Eigen::VectorXd::Zero(x.size());
        for (const auto& subsystem_cost : subsystem_cost_functions) {
            grad_x += subsystem_cost->gradient_x(x, u);
        }
        return grad_x;
    }

    Eigen::VectorXd gradient_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        Eigen::VectorXd grad_u = Eigen::VectorXd::Zero(u.size());
        for (const auto& subsystem_cost: subsystem_cost_functions) {
            grad_u += subsystem_cost->gradient_u(x, u);
        }
        return grad_u;
    }

    Eigen::MatrixXd hessian_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        Eigen::MatrixXd hessian_x_matrix = Eigen::MatrixXd::Zero(x.size(), x.size());
        for (const auto& subsystem_cost : subsystem_cost_functions) {
            hessian_x_matrix += subsystem_cost->hessian_x(x, u);
        }
        return hessian_x_matrix;
    }

    Eigen::MatrixXd hessian_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        Eigen::MatrixXd hessian_u_matrix = Eigen::MatrixXd::Zero(u.size(), u.size());
        for (const auto& subsystem_cost : subsystem_cost_functions) {
            hessian_u_matrix += subsystem_cost->hessian_u(x, u);
        }
        return hessian_u_matrix;
    }
};

int main() {
    // Example usage
    VectorXd x_ref(8);
    x_ref << 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0;

    std::shared_ptr<ProximityCost> proximity_cost = std::make_shared<ProximityCost>(0.8, 0, 1, 1.0);
    std::shared_ptr<ReferenceCost> reference_cost = std::make_shared<ReferenceCost>(0, x_ref, 1.0);
    std::shared_ptr<WallCost> wall_cost = std::make_shared<WallCost>(0, 1.0);
    std::shared_ptr<InputCost> input_cost = std::make_shared<InputCost>(5.0, 3.0);

     // Declare and initialize x
    Eigen::VectorXd x_example(8); // Declare and initialize x_example
    Eigen::VectorXd u_example(2); // Declare and initialize u_example

    x_example << 1.0, 2.0, 3.0, 4.0, 1.1, 2.3, 7.0, 3.5;
    u_example << 9.0, 10.0;

 
    std::vector<std::shared_ptr<CostBase>> all_costs = {proximity_cost, reference_cost, wall_cost, input_cost};


    OverallCost overall_cost_all(all_costs);
    double total_cost_all = overall_cost_all.evaluate(x_example, u_example);
    Eigen::VectorXd gradient_x_all = overall_cost_all.gradient_x(x_example, u_example);
    Eigen::MatrixXd hessian_x_all = overall_cost_all.hessian_x(x_example, u_example);
    Eigen::MatrixXd hessian_u_all = overall_cost_all.hessian_u(x_example, u_example);


    std::cout << "Total cost for all costs: " << total_cost_all << std::endl;
    std::cout << "Gradient w.r.t. x for all costs: " << gradient_x_all << std::endl;
    std::cout << "Hessian w.r.t. x for all costs: " << hessian_x_all << std::endl;
    std::cout << "Hessian w.r.t. u for all costs: " << hessian_u_all << std::endl;

    return 0;
}

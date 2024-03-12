#include <iostream>
#include <C:\Program Files\Eigen3\Eigen\Dense>
#include <memory>

using namespace Eigen;


class ProximityCost {
public:
    double d_threshold;
    int idx1, idx2;
    double weight;

    ProximityCost(double d_threshold = 0.5, int idx1 = 0, int idx2 = 0, double weight = 1.0)
        : d_threshold(d_threshold), idx1(idx1), idx2(idx2), weight(weight) {}

    double evaluate(const VectorXd& x, const VectorXd& u) const {
        double dist = std::sqrt(std::pow(x[4 * idx1] - x[4 * idx2], 2) + std::pow(x[4 * idx1 + 1] - x[4 * idx2 + 1], 2));
        return (dist > d_threshold) ? 0.0 : weight * (d_threshold - dist);
    }

    VectorXd gradient_x(const VectorXd& x, const VectorXd& u) const {
        double dist = std::sqrt(std::pow(x[4 * idx1] - x[4 * idx2], 2) + std::pow(x[4 * idx1 + 1] - x[4 * idx2 + 1], 2));
        if (dist > d_threshold) {
            return VectorXd::Zero(x.size());
        }

        double denom = -weight / (2 * std::sqrt(std::pow(x[4 * idx1] - x[4 * idx2], 2) + std::pow(x[4 * idx1 + 1] - x[4 * idx2 + 1], 2)) + 1e-6);

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
};

class ReferenceCost {
public:
    int idx;
    VectorXd x_ref;
    double weight;

    ReferenceCost(int idx = 0, const VectorXd& x_ref = VectorXd::Zero(8), double weight = 1.0)
        : idx(idx), x_ref(x_ref), weight(weight) {}

    double evaluate(const VectorXd& x, const VectorXd& u) const {
        double dist = std::sqrt(std::pow(x[4 * idx] - x_ref[4 * idx], 2) +
                                std::pow(x[4 * idx + 1] - x_ref[4 * idx + 1], 2) +
                                std::pow(x[4 * idx + 2] - x_ref[4 * idx + 2], 2) +
                                std::pow(x[4 * idx + 3] - x_ref[4 * idx + 3], 2));
        return dist * weight;
    }

    VectorXd gradient_x(const VectorXd& x, const VectorXd& u) const {
        double denom = weight / (2 * std::sqrt(std::pow(x[4 * idx] - x_ref[4 * idx], 2) +
                                              std::pow(x[4 * idx + 1] - x_ref[4 * idx + 1], 2) +
                                              std::pow(x[4 * idx + 2] - x_ref[4 * idx + 2], 2) +
                                              std::pow(x[4 * idx + 3] - x_ref[4 * idx + 3], 2)));

        VectorXd grad_x = VectorXd::Zero(x.size());
        grad_x[4 * idx] = 2 * (x[4 * idx] - x_ref[4 * idx]) * denom;
        grad_x[4 * idx + 1] = 2 * (x[4 * idx + 1] - x_ref[4 * idx + 1]) * denom;
        grad_x[4 * idx + 2] = 2 * (x[4 * idx + 2] - x_ref[4 * idx + 2]) * denom;
        grad_x[4 * idx + 3] = 2 * (x[4 * idx + 3] - x_ref[4 * idx + 3]) * denom;

        return grad_x;
    }

    VectorXd gradient_u(const VectorXd& x, const VectorXd& u) const {
        return VectorXd::Zero(u.size());
    }
};

class InputCost {
public:
    int idx;
    double weight;

    InputCost(int idx, double weight = 1.0) : idx(idx), weight(weight) {}

    double evaluate(const VectorXd& x, const VectorXd& u) const {
        return weight * (5 * std::pow(u[0], 2) + 7 * std::pow(u[1], 2));
    }

    VectorXd gradient_x(const VectorXd& x, const VectorXd& u) const {
        return VectorXd::Zero(x.size());
    }

    VectorXd gradient_u(const VectorXd& x, const VectorXd& u) const {
        VectorXd grad_u = VectorXd::Zero(u.size());
        grad_u[0] = 2 * weight * u[0] * 5;
        grad_u[1] = 2 * weight * u[1] * 7;
        return grad_u;
    }
};

class WallCost {
public CostBase {:
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
        VectorXd grad_x = VectorXd::Zero(x.size());

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
}};

class TrialCost {
public:
    double d_threshold;

    TrialCost(double d_threshold = 0.5) : d_threshold(d_threshold) {}

    double evaluate(const VectorXd& x, const VectorXd& u) const {
        double dist = std::pow(x[0], 2) + std::pow(x[1], 2) + std::pow(x[2], 2) + std::pow(x[3], 2) +
                      std::pow(x[4], 2) + std::pow(x[5], 2) + std::pow(x[6], 2) + std::pow(x[7], 2);
        return dist;
    }
};

class CostBase {
public:
    virtual double evaluate(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const = 0;
    virtual Eigen::VectorXd gradient_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const = 0;
    virtual Eigen::VectorXd gradient_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const = 0;
    virtual Eigen::MatrixXd hessian_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const = 0;
    virtual Eigen::MatrixXd hessian_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const = 0;
    // Add any other necessary methods or members.
};

class OverallCost {
public:
    std::vector<std::shared_ptr<CostBase>> subsystem_cost_functions;

    OverallCost(const std::vector<std::shared_ptr<CostBase>>& subsystem_cost_functions) {
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
        for (const auto& subsystem_cost : subsystem_cost_functions) {
            grad_u += subsystem_cost->gradient_u(x, u);
        }
        return grad_u;
    }

    Eigen::MatrixXd hessian_x(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        Eigen::MatrixXd hessian_x = Eigen::MatrixXd::Zero(x.size(), x.size());
        for (const auto& subsystem_cost : subsystem_cost_functions) {
            hessian_x += subsystem_cost->hessian_x(x, u);
        }
        return hessian_x;
    }

    Eigen::MatrixXd hessian_u(const Eigen::VectorXd& x, const Eigen::VectorXd& u) const {
        Eigen::MatrixXd hessian_u = Eigen::MatrixXd::Zero(u.size(), u.size());
        for (const auto& subsystem_cost : subsystem_cost_functions) {
            hessian_u += subsystem_cost->hessian_u(x, u);
        }
        return hessian_u;
    }
};
int main() {
    // Example usage
    std::shared_ptr<ProximityCost> proximity_cost = std::make_shared<ProximityCost>(0.5, 0, 1, 1.0);
    std::shared_ptr<ReferenceCost> reference_cost = std::make_shared<ReferenceCost>(0, VectorXd::Zero(8), 1.0);
    std::shared_ptr<WallCost> wall_cost = std::make_shared<WallCost>(0, 1.0);

    std::vector<std::shared_ptr<ProximityCost>> proximity_costs = {proximity_cost};
    std::vector<std::shared_ptr<ReferenceCost>> reference_costs = {reference_cost};
    std::vector<std::shared_ptr<WallCost>> wall_costs = {wall_cost};

    OverallCost overall_cost_proximity(proximity_costs);
    OverallCost overall_cost_reference(reference_costs);
    OverallCost overall_cost_wall(wall_costs);

    VectorXd x_example(8);
    x_example << -8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    VectorXd u_example(2);
    u_example << 1, 1;

    double total_cost_proximity = overall_cost_proximity.evaluate(x_example, u_example);
    VectorXd gradient_x_proximity = overall_cost_proximity.gradient_x(x_example, u_example);
    MatrixXd hessian_x_proximity = overall_cost_proximity.hessian_x(x_example, u_example);
    MatrixXd hessian_u_proximity = overall_cost_proximity.hessian_u(x_example, u_example);

    double total_cost_reference = overall_cost_reference.evaluate(x_example, u_example);
    VectorXd gradient_x_reference = overall_cost_reference.gradient_x(x_example, u_example);
    MatrixXd hessian_x_reference = overall_cost_reference.hessian_x(x_example, u_example);
    MatrixXd hessian_u_reference = overall_cost_reference.hessian_u(x_example, u_example);

    double total_cost_wall = overall_cost_wall.evaluate(x_example, u_example);
    VectorXd gradient_x_wall = overall_cost_wall.gradient_x(x_example, u_example);
    MatrixXd hessian_x_wall = overall_cost_wall.hessian_x(x_example, u_example);
    MatrixXd hessian_u_wall = overall_cost_wall.hessian_u(x_example, u_example);

    std::cout << "Total Cost (Proximity): " << total_cost_proximity << std::endl;
    std::cout << "Gradient with respect to x (Proximity): " << gradient_x_proximity.transpose() << std::endl;
    std::cout << "Hessian with respect to x (Proximity):\n" << hessian_x_proximity << std::endl;
    std::cout << "Hessian with respect to u (Proximity):\n" << hessian_u_proximity << std::endl;

    std::cout << "Total Cost (Reference): " << total_cost_reference << std::endl;
    std::cout << "Gradient with respect to x (Reference): " << gradient_x_reference.transpose() << std::endl;
    std::cout << "Hessian with respect to x (Reference):\n" << hessian_x_reference << std::endl;
    std::cout << "Hessian with respect to u (Reference):\n" << hessian_u_reference << std::endl;

    std::cout << "Total Cost (Wall): " << total_cost_wall << std::endl;
    std::cout << "Gradient with respect to x (Wall): " << gradient_x_wall.transpose() << std::endl;
    std::cout << "Hessian with respect to x (Wall):\n" << hessian_x_wall << std::endl;
    std::cout << "Hessian with respect to u (Wall):\n" << hessian_u_wall << std::endl;

    return 0;
}

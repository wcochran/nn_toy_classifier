#include <vector>
#include <memory>
#include <cassert>
#include <Eigen/Dense>

class Module {
    const size_t _numInputs, _numOutputs;
public:
    Module(size_t in, size_t out) : _numInputs{in}, _numOutputs{out} {}
    virtual ~Module() {};
    size_t numInputs() const {return _numInputs;}
    size_t numOutputs() const {return _numOutputs;}
    virtual const Eigen::VectorXf& operator()(const Eigen::VectorXf& input) = 0;
    virtual const Eigen::VectorXf& output() const = 0;
};

class LinearModule : public Module { // fully connected
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> _W;
    Eigen::Vector<float, Eigen::Dynamic> _b;
    Eigen::Vector<float, Eigen::Dynamic> _z;
public:
    LinearModule(size_t in, size_t out)
        : Module(in, out), _W(out,in), _b(out), _z(out) {}
    virtual const Eigen::VectorXf& operator()(const Eigen::VectorXf& x) {
        assert(x.rows() == numInputs());
        _z.noalias() = _W*x + _b;
        return _z;
    }
    virtual const Eigen::VectorXf& output() const {return _z;}
};

class ReLUModule : public Module {
    Eigen::Vector<float, Eigen::Dynamic> _a;
public:
    ReLUModule(size_t inOut) : Module(inOut, inOut), _a(inOut) {}
    virtual const Eigen::VectorXf& operator()(const Eigen::VectorXf& x) {
        _a.noalias() = x.unaryExpr([](float elem) -> float {
            return std::max(0.f,elem);
        });
        return _a;
    }
    virtual const Eigen::VectorXf& output() const {return _a;}
};

class SoftMaxModule : public Module {
    Eigen::Vector<float, Eigen::Dynamic> _a;
public:
    SoftMaxModule(size_t inOut) : Module(inOut, inOut), _a(inOut) {}
    virtual const Eigen::VectorXf& operator()(const Eigen::VectorXf& x) {
        assert(x.rows() == numInputs());
        const Eigen::VectorXf e2x = x.unaryExpr([](float elem) -> float {
            return std::exp(elem);
        });
        const float sum = e2x.sum();
        _a.noalias() = e2x.unaryExpr([sum](float elem) -> float {
            return elem/sum;
        });
        return _a;
    }
    virtual const Eigen::VectorXf& output() const {return _a;}
};

class SequentialModule : public Module {
    std::vector<std::unique_ptr<Module>> _modules;
public:
    SequentialModule(std::vector<std::unique_ptr<Module>>&& modules)
    : Module(modules.front()->numInputs(), modules.back()->numOutputs()),
      _modules(std::move(modules)) {}

    virtual const Eigen::VectorXf& operator()(const Eigen::VectorXf& x) {
        (*_modules[0])(x);
        for (size_t i = 1; i < _modules.size(); i++) {
            (*_modules[i])(_modules[i-1]->output());
        }
        return _modules.back()->output();
    }
    virtual const Eigen::VectorXf& output() const {
        return _modules.back()->output();
    }
};

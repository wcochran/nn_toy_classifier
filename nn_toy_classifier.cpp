#include <vector>
#include <memory>
#include <cassert>
#include <Eigen/Dense>

class Module {
    enum class ModelType {
        Linear,
        Sigmoid,
        ReLU,
        SoftMax,
        Sequential,
        MSELoss
    };
    ModuleType _type;
    const size_t _numInputs, _numOutputs;
    Model *_prev;  // weak ptr to previous layer (nullptr => input layer)
public:
    Module(ModuleType t, size_t in, size_t out) : _type{t}, _numInputs{in}, _numOutputs{out}, _prev{nullptr} {}
    virtual ~Module() {};
    size_t numInputs() const {return _numInputs;}
    size_t numOutputs() const {return _numOutputs;}
    virtual const Eigen::VectorXf& operator()(const Eigen::VectorXf& input) = 0;
    virtual const Eigen::VectorXf& output() const = 0;

    void setPreviousLayer(Model* prev) {_prev = prev;}
    virtual void zeroGrad() {};
    virtual void backward(const Eigen::VectorXf& grad) = 0;
    virtual void update(float learningRate) {};
};

class LinearModule : public Module { // fully connected
    Eigen::VectorXf _x;   // cached input (+ homogenous 1 added)
    Eigen::MatrixXf _W;   // weights + bias
    Eigen::VectorXf _z;   // cached output
    Eigen::MatrixXf _dW;  // weight + bias gradients
public:
    LinearModule(size_t in, size_t out)
        : Module(Linear, in, out), _x(in), _W(out,in+1), _z(out), _dW(out,in+1) {}
    
    virtual const Eigen::VectorXf& operator()(const Eigen::VectorXf& x) {
        assert(x.rows() == numInputs());
        _x = x.homogeneous();
        _z.noalias() = _W * _x;
        return _z;
    }
    virtual const Eigen::VectorXf& output() const {return _z;}

    virtual void zeroGrad() { _dW.setZero(); }

    //
    // We are given the input accumulated gradients {dL/dz_i} for each
    // of our outputs z_i which have propagated backwards to the
    // output of this layer.
    // N = number of outputs
    // M = number of inputs
    // i = index for output layer z_i  (i = 0 .. N-1)
    // j = index for input later x_j   (j = 0 .. M-1)
    // W is N x M  [ W_ij ]
    // For updating our local parameter gradients we need:
    //       dL/dW_{ij} = dL/dz_i * dz_i/dW_{ij} = dL/dz_i * x_j
    //                     _           _   _                    _
    //       dL/dW      = |   dL/dz_0   | | x_0  x_1 ... x_{M-1} |
    //                    |   dL/dz_1   |  -                    -
    //                    |      .      |
    //                    | dL/dz_{N-1} |  (note: no dependence on W)
    //                     -           _
    // For continued back propogation to the prevous layer we need:
    //       dL/dx_j = Sum_i dL/dz_i * dz_i/dx_j
    //               = Sum_i dL/dz_i * W_{ij}
    //       dL/dx   = [dL/dz_0 dL/dz_1 ... dL/dz_{N-1}] * W
    //
    virtual void backward(const Eigen::VectorXf& dz) {
        assert(dz.size() = numOutputs());
        const Eigen::MatrixXf dW = dz * _x.transpose();
        _dW += dW;
        if (_prev != nullptr) {
            const Eigen::Vector dx = dz.transpose() * W;
            _prev->backward(dx);
        }
    }
    
    virtual void update(float learningRate) {
        _W -= learningRate * _dW; // step in negative dir of gradient
    }
};

class SigmoidModule : public Module {
    Eigen::VectorXf _z; // cached output
    ReLUModule(size_t inOut) : Module(Sigmoid, inOut, inOut), _a(inOut) {}
public:
    virtual const Eigen::VectorXf& operator()(const Eigen::VectorXf& x) {
        _z.noalias() = x.unaryExpr([](float elem) -> float {
            const float ex = std::exp(elem);
            return ex/(ex + 1);
        });
        return _z;
    }
    virtual const Eigen::VectorXf& output() const {return _z;}
    virtual void backward(const Eigen::VectorXf& grad) {
        if (_prev != nullptr) {
            assert(grad.size() == numOutputs());
            const Eigen::VectorXf one = Eigen::VectorXf::Constant(numOutputs,1.0f);
            const Eigen::VectorXf dzdx = _z.cwizeProdect*(one - _z); // s' = s*(1 - s)
            const Eigen::VectorXf dLdz = grad.cwiseProduct(dzdx);
            _prev->backward(dLdZ);
        }
    }
};

class ReLUModule : public Module {
    Eigen::VectorXf _a; // cached output
public:
    ReLUModule(size_t inOut) : Module(ReLU, inOut, inOut), _a(inOut) {}
    virtual const Eigen::VectorXf& operator()(const Eigen::VectorXf& x) {
        _a.noalias() = x.unaryExpr([](float elem) -> float {
            return std::max(0.f,elem);
        });
        return _a;
    }
    virtual const Eigen::VectorXf& output() const {return _a;}

    virtual void backward(const Eigen::VectorXf& grad) {
        if (_prev != nullptr) {
            assert(grad.size() == numOutputs());
            const Eigen::VectorXf dzdx = _a.unaryExpr([](float a) -> float {
                return (a > 0) ? 1.0f : 0.0f;
            });
            const Eigen::VectorXf dLdz = grad.cwiseProduct(dzdx);
            _prev->backward(dLdZ);
        }
    }
};

class SoftMaxModule : public Module {
    Eigen::Vector<float, Eigen::Dynamic> _a;
public:
    SoftMaxModule(size_t inOut) : Module(SoftMax, inOut, inOut), _a(inOut) {}
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
    virtual void backward(const Eigen::VectorXf& grad) {
        assert(false); // unimplemented
    }
};

class SequentialModule : public Module {
    std::vector<std::unique_ptr<Module>> _modules;
public:
    SequentialModule(std::vector<std::unique_ptr<Module>>&& modules)
        : Module(Sequential, modules.front()->numInputs(), modules.back()->numOutputs()),
          _modules(std::move(modules)) {
        for (size_t i = 1; i < _modules.size(); i++)
            _modules[i].setPrevious(_modules[i-1].get());
    }

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

    virtual void zeroGrad() {
        for (auto&& m : _modules)
            m->zeroGrad();
    }
    
    virtual void backward(const Eigen::VectorXf& grad) {
        assert(grad.size() == _modules.back().numOutputs());
        _modules.back()->backward(grad);
    }
    
    virtual void update(float learningRate) {
        for (size_t i = 0; i < _modules.size(); i++)
            _modules[i]->update(learningRate);
    }
};

class MSELossModule : public Module {
    Eigen::VectorXf _x, _y; // cached input
    float _loss;            // cached output
public:
    MSELossModule(size_t in) : Module(MSELoss, in, 1) {}
    virtual const Eigen::VectorXf& operator()(const Eigen::VectorXf& x) {
        assert(false);  // not used for loss functions
        return Eigen::VectorXf::Zero(1); // not reached;
    }
    virtual const float operator()(const Eigen::VectorXf& x,
                                   const Eigen::VectorYf& y) {
        _x = x; // computed result
        _y = y; // target result
        _loss = 0.5 * (x - y).squaredNorm();
        return _loss;
    }
    virtual void backward(const Eigen::VectorXf& dz) {
        assert(false);
    }
    virtual void backward() {
        if (_prev == nullptr) return;
        const Eigen::VectorXf dz = _x - _y;
        _prev->backward(dz);
    }
};
    
int main(int argc, char *argv[]) {
    // SequentialModule classifier
    // (
    //  {
    //      std::make_unique<LinearModule>(782, 128),
    //      std::make_unique<ReLUModule>(128),
    //      std::make_unique<LinearModule>(128, 64),
    //      std::make_unique<ReLUModule>(64),
    //      std::make_unique<LinearModule>(64, 10),
    //      std::make_unique<SoftMaxModule>(10)
    //  }
    // );

    std::vector<std::unique_ptr<Module>> modules;
    modules.emplace_back(std::make_unique<LinearModule>(782, 128));
    modules.emplace_back(std::make_unique<ReLUModule>(128));
    modules.emplace_back(std::make_unique<LinearModule>(128, 64));
    modules.emplace_back(std::make_unique<ReLUModule>(64));
    modules.emplace_back(std::make_unique<LinearModule>(64, 10));
    modules.emplace_back(std::make_unique<SoftMaxModule>(10));
    SequentialModule classifier(std::move(modules));
    
    return 0;
}

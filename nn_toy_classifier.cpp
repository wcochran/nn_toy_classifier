#include <vector>
#include <memory>
#include <cassert>
#include <Eigen/Dense>
#include <random>
#include <functional>

class Module {
public:
    enum class ModuleType {
        Linear,
        Sigmoid,
        ReLU,
        SoftMax,
        Sequential,
        MSELoss
    };
    const ModuleType _type;
    const size_t _numInputs, _numOutputs;
    Module *_prev;  // weak ptr to previous layer (nullptr => input layer)
    Module(ModuleType t, size_t in, size_t out) : _type{t}, _numInputs{in}, _numOutputs{out}, _prev{nullptr} {}
    virtual ~Module() {};
    size_t numInputs() const {return _numInputs;}
    size_t numOutputs() const {return _numOutputs;}

    //
    // * Forward pass operates on batch input. Each column of the matrix
    //   represents a single input vector. We will know the number of inputs
    //   when the Module is constructed, but the batch size may be different
    //   for each batch of input. Therefore some modules will likely have to dynamically
    //   resize matrices that cache data.
    // * The input data is assumed to persist for the lifetime of a forward and
    //   and backward pass so that it can be interogated when gradients are
    //   computed. We avoid
    // * Linear modules will want their input vectors to be homogenous.
    //   Unfortunatly they will have to make a copy on the input to add
    //   the extra row of 1's. Eigen doesn't seem to provide a multiplication
    //   operation that assumes an implicit 1 added (?)
    //
    void operator()(Eigen::MatrixXf&& input) = delete; // dot no pass temporary
    virtual const Eigen::MatrixXf& operator()(const Eigen::MatrixXf& input) = 0;
    virtual const Eigen::MatrixXf& output() const = 0;

    //
    // Backward pass receives a batch of gradient values from the successive
    // layer -- each column is one vector of derivatives where each element
    // is an input to the corresponding output node of the current layer.
    //
    void setPrevious(Module* prev) {_prev = prev;}
    virtual void zeroGrad() {};
    virtual void backward(const Eigen::MatrixXf& grad) = 0;
    virtual void update(float learningRate) {};
};

class LinearModule : public Module { // fully connected
    Eigen::MatrixXf _X;   // copy of last input (+ homogenous 1 added)
    Eigen::MatrixXf _W;   // weights + bias
    Eigen::MatrixXf _Z;   // cached output
    Eigen::MatrixXf _dW;  // weight + bias gradients
    size_t _numGradients; // number of accumulate gradients in _dW
public:
    LinearModule(size_t in, size_t out)
        : Module(ModuleType::Linear, in, out), _X(in+1,1), _W(out,in+1), _Z(out,1),
          _dW(out,in+1), _numGradients{0} {
              _X(in) = 1;
          }
    
    virtual const Eigen::MatrixXf& operator()(const Eigen::MatrixXf& X) {
        assert(X.rows() == numInputs());
        // may want to check sizes of _X and _Z and use noalias() if they are the right size
        _X = X.colwise().homogeneous();  // make copy of input with added row of 1's
        _Z = _W * _X;
        return _Z;
    }
    virtual const Eigen::MatrixXf& output() const {return _Z;}

    //
    // The last column are the biases
    //
    Eigen::MatrixXf& weights() { return _W; }
    
    virtual void zeroGrad() {
        _numGradients = 0;
        _dW.setZero();
    }

    //
    //  We saved the corresponding inputs and outputs in _X and _Z.
    //     _Z = _W * _X
    //  Let M = num inputs, N = num outputs, B = batch size.
    //  We are give the gradients of the loss function with respect to output i
    //  for each batch item k:
    //     [ dZ_{ik} ] = [ dL/dz_{ik} ],  i = 0 .. N-1,  k = 0 .. N-1
    //
    //  To update out local weights we need the following gradients for batch k
    //     dL/W_{ij} = dL/dz_{ik} * dz_{ik}/dW_{ij}
    //  We have
    //     dz_{ik}/dW_{ij} = x_{jk}
    //  therefore for batch k we have
    //     dL/W_{ij} = dL/dz_{ik} * x_{jk}.
    //  We will sum all the gradients over all the batches
    //     dL/W_{ij} = Sum_{k=0..B-1} dL/dz_{ik} * x_{jk}.
    //               = dZ * X^T
    //
    //  For continued back propogation to the previous layer (our inputs X are
    //  the previous layer's outputs) we need the
    //     dL/dX_{jk} = Sum_{i=0..N-1} dL/dz_{ik} * dZ_{ik}/dX_{jk}
    //  We have
    //     dZ_{ik}/dX_{jk} = W_{ij}  (same gradient for all batches)
    //  therefore
    //     dL/dX_{jk} = Sum_{i=0..N-1} dL/dz_{ik} * W_{ij}
    //                = dZ^T * W
    //
    virtual void backward(const Eigen::MatrixXf& dZ) {
        assert(dZ.rows() == numOutputs());
        assert(dZ.cols() == _X.cols());  // batch size
        const Eigen::MatrixXf dW = dZ * _X.transpose();
        _dW += dW;
        _numGradients += _X.cols();
        if (_prev != nullptr) {
            const Eigen::MatrixXf dX = dZ.transpose() * _W;
            _prev->backward(dX);
        }
    }
    
    virtual void update(float learningRate) {
        assert(_numGradients > 0);
        _W -= learningRate / _numGradients * _dW; // step in negative dir of gradient
    }
};

//
// Sigmoid activation function.
//   s(x) = e^x/(1 + e^x);
//   s'(x) = s(x)*(1 - s(x))
//
class SigmoidModule : public Module {
    Eigen::MatrixXf _Z; // cached output
    SigmoidModule(size_t inOut) : Module(ModuleType::Sigmoid, inOut, inOut), _Z(inOut,1) {}
public:
    virtual const Eigen::MatrixXf& operator()(const Eigen::MatrixXf& X) {
        _Z = X.unaryExpr([](float elem) -> float {
            const float ex = std::exp(elem);
            return ex/(ex + 1);
        });
        return _Z;
    }
    virtual const Eigen::MatrixXf& output() const {return _Z;}
    virtual void backward(const Eigen::MatrixXf& dLdZ) {
        if (_prev != nullptr) {
            const auto N = numOutputs();
            const auto B = _Z.cols(); // batch size
            assert(dLdZ.rows() == N);
            assert(dLdZ.cols() == B);
            const Eigen::MatrixXf One = Eigen::MatrixXf::Constant(N, B, 1.0f);
            const Eigen::VectorXf dZdX = _Z.array() * (One - _Z).array(); // s' = s*(1 - s)
            const Eigen::VectorXf dLdX = dLdZ * dZdX;
            _prev->backward(dLdX);
        }
    }
};

class ReLUModule : public Module {
    Eigen::MatrixXf _Z; // cached output
public:
    ReLUModule(size_t inOut) : Module(ModuleType::ReLU, inOut, inOut), _Z(inOut,1) {}
    virtual const Eigen::MatrixXf& operator()(const Eigen::MatrixXf& X) {
        _Z = X.unaryExpr([](float elem) -> float {
            return std::max(0.f,elem);
        });
        return _Z;
    }
    virtual const Eigen::MatrixXf& output() const {return _Z;}

    virtual void backward(const Eigen::VectorXf& dLdZ) {
        if (_prev != nullptr) {
            assert(dLdZ.rows() == numOutputs());
            assert(dLdZ.cols() == _Z.cols());
            const Eigen::MatrixXf dZdX = _Z.unaryExpr([](float z) -> float {
                return (z > 0) ? 1.0f : 0.0f;
            });
            const Eigen::VectorXf dLdX = dLdZ.array() * dZdX.array();
            _prev->backward(dLdX);
        }
    }
};

//class SoftMaxModule : public Module {
//    Eigen::Vector<float, Eigen::Dynamic> _a;
//public:
//    SoftMaxModule(size_t inOut) : Module(ModuleType::SoftMax, inOut, inOut), _a(inOut) {}
//    virtual const Eigen::VectorXf& operator()(const Eigen::VectorXf& x) {
//        assert(x.rows() == numInputs());
//        const Eigen::VectorXf e2x = x.unaryExpr([](float elem) -> float {
//            return std::exp(elem);
//        });
//        const float sum = e2x.sum();
//        _a.noalias() = e2x.unaryExpr([sum](float elem) -> float {
//            return elem/sum;
//        });
//        return _a;
//    }
//    virtual const Eigen::VectorXf& output() const {return _a;}
//    virtual void backward(const Eigen::VectorXf& grad) {
//        assert(false); // unimplemented
//    }
//};

class SequentialModule : public Module {
    std::vector<std::shared_ptr<Module>> _modules;
public:
    SequentialModule(std::vector<std::shared_ptr<Module>>& modules)
        : Module(ModuleType::Sequential, modules.front()->numInputs(), modules.back()->numOutputs()),
          _modules(modules) {
        for (size_t i = 1; i < _modules.size(); i++)
            _modules[i]->setPrevious(_modules[i-1].get());
    }

    virtual const Eigen::MatrixXf& operator()(const Eigen::MatrixXf& X) {
        (*_modules[0])(X);
        for (size_t i = 1; i < _modules.size(); i++) {
            (*_modules[i])(_modules[i-1]->output());
        }
        return _modules.back()->output();
    }
    virtual const Eigen::MatrixXf& output() const {
        return _modules.back()->output();
    }

    virtual void zeroGrad() {
        for (auto&& m : _modules)
            m->zeroGrad();
    }
    
    virtual void backward(const Eigen::MatrixXf& grad) {
        assert(grad.rows() == _modules.back()->numOutputs());
        _modules.back()->backward(grad);
    }
    
    virtual void update(float learningRate) {
        for (size_t i = 0; i < _modules.size(); i++)
            _modules[i]->update(learningRate);
    }
};

class MSELossModule : public Module {
    Eigen::MatrixXf _X, _Y; // cached input
    Eigen::MatrixXf _loss;  // cached output
public:
    MSELossModule(size_t in) :
        Module(ModuleType::MSELoss, in, 1), _X(in,1), _Y(in,1), _loss(1,1) {}
    virtual const Eigen::MatrixXf& operator()(const Eigen::MatrixXf& X) {
        assert(false);  // not used for loss functions
        static Eigen::MatrixXf nada = Eigen::MatrixXf::Zero(1,1);
        return nada;
    }
    virtual const Eigen::MatrixXf operator()(const Eigen::MatrixXf& X,
                                             const Eigen::MatrixXf& Y) {
        _X = X; // computed result
        _Y = Y; // target result
        _loss = 0.5 * (X - Y).colwise().squaredNorm();
        return _loss;
    }
    virtual const Eigen::MatrixXf& output() const { return _loss; }
    virtual void backward(const Eigen::MatrixXf& dz) {
        assert(false);
    }
    virtual void backward() {
        if (_prev == nullptr) return;
        const Eigen::MatrixXf dZ = _X - _Y;
        _prev->backward(dZ);
    }
};

void kaimingNormalInit(Eigen::MatrixXf& W) {
    const auto N = W.cols() - 1; // num inputs
    const auto M = W.rows();     // num outputs
    const float var = 2.0/(N + M);
    std::mt19937 gen(1234);
    std::normal_distribution<> dist{0.0f, var};
    for (size_t i = 0; i < M; i++)
        for (size_t j = 0; j < N; j++)
            W(i,j) = dist(gen);
    W.col(N) = Eigen::VectorXf::Zero(M); // biases
}

#include <iostream>

void celcius_to_farenheit() {

    //
    // Create some input values and labels.
    //
    constexpr size_t n = 10;
    constexpr float lo_c = 0, hi_c = 100;
    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dist(lo_c, hi_c);
    Eigen::MatrixXf F(1,n);
    Eigen::MatrixXf C(1,n);
    for (size_t i = 0; i < n; i++) {
        const float c = dist(gen);
        const float f = 9.0f/5 * c + 32.0f;
        C(0,i) = c;
        F(0,i) = f;
    }

    //
    // Normalize input using mean and stdev of
    // uniform distribution on interval [a,b].
    //
    constexpr float c_mean = (lo_c + hi_c)/2;
    constexpr float sqrt12 = 3.4641016151f;
    constexpr float c_stdev = (hi_c - lo_c)/sqrt12;
    Eigen::MatrixXf Cnorm(1,n);
    for (size_t i = 0; i < n; i++) {
        Cnorm(0,i) = (C(0,i) - c_mean)/c_stdev;
    }

    //
    // Create MLP with one linear layer.
    //
    std::vector<std::shared_ptr<Module>> modules;
    auto linearLayer = std::make_shared<LinearModule>(1, 1);
    modules.emplace_back(linearLayer);
    SequentialModule converter(modules);

    //
    // Create loss function
    //
    MSELossModule lossFunc(converter.numOutputs());
    lossFunc.setPrevious(&converter);

    //
    // Initialize linear module weights according to
    // Kaiming He algorithm (mimic torch.nn.init.kaiming_normal)
    // This is typically used when the linear layer is followed
    // by a ReLU function (which we don't have).
    //
    kaimingNormalInit(linearLayer->weights());

    //
    // Train.
    // Batch size = Input size.
    //
    constexpr size_t numEpochs = 70;
    constexpr float learningRate = 0.5;
    for (size_t epoch = 0; epoch < numEpochs; epoch++) {
        converter.zeroGrad();
        const Eigen::MatrixXf y = converter(Cnorm);
        const Eigen::MatrixXf loss = lossFunc(y,F);
        const float totalLoss = loss.sum();
        lossFunc.backward();
        const float averageLoss = totalLoss / n;
        std::cout << "epoch:" << epoch << ": loss = " << averageLoss << "\n";
        converter.update(learningRate);
    }

    //
    // Trained celsius to farenheit converter.
    //
    auto celciusToFarenhright = [&](float celcius) -> float {
        const float cnorm = (celcius - c_mean)/c_stdev;
        Eigen::VectorXf cvec(1);
        cvec(0) = cnorm;
        const Eigen::VectorXf y = converter(cvec);
        return y(0);
    };

    //
    // Validation with known truth.
    //
    constexpr size_t nsamples = 20;
    const float dc = (hi_c - lo_c)/(nsamples - 1);
    for (size_t i = 0; i < nsamples; i++) {
        const float c = lo_c + i*dc;
        const float f = celciusToFarenhright(c);
        const float f_truth = 9.0/5 * c + 32;
        const float error = std::fabs(f - f_truth);
        std::cout << c << " " << f << " " << error << "\n";
    }

}

int main(int argc, char *argv[]) {
    celcius_to_farenheit();
    return 0;
}


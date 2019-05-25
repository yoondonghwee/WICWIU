#include "../Module.hpp"

template<typename DTYPE> class RNN : public Module<DTYPE>{
private:
public:
    // 생성자 만들꺼지
    RNN(Operator<DTYPE> *pInput, Operator<DTYPE> *hidden, int pNumInputCol, int pNumOutputCol, int use_bias = TRUE, std::string pName = "No Name") : Module<DTYPE>(pName) {
        Alloc(pInput, hidden, pNumInputCol, pNumOutputCol, use_bias, pName);
    }

    virtual ~RNN() {}


    int Alloc(Operator<DTYPE> *pInput, Operator<DTYPE> *hidden, int pNumInputCol, int pNumOutputCol, int use_bias, std::string pName) {
        this->SetInput(pInput);

        Operator<DTYPE> *out = pInput;
        Operator<DTYPE> *out1 = NULL, out2 = NULL, out3 = NULL, out4 = NULL, out5 = NULL, hidden_next = NULL;

        float stddev = 0.1;

        Tensorholder<DTYPE> *pWeight_xh = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, pNumOutputCol, pNumInputCol, 0.0, stddev), "RNN_Weight_xh_" + pName);
        out1 = new MatMul<DTYPE>(pWeight_xh, out, "RNN_MatMul_xh_" + pName);

        Tensorholder<DTYPE> *pWeight_hh = new Tensorholder<DTYPE>(Tensor<DTYPE>::Random_normal(1, 1, 1, pNumOutputCol, pNumOutputCol, 0.0, stddev), "RNN_Weight_hh_" + pName);
        out2 = new MatMul<DTYPE>(pWeight_hh, hidden, "RNN_MatMul_hh_" + pName);

        if (use_bias) {
            Tensorholder<DTYPE> *pBias_xh = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, pNumOutputCol, 0.f), "Add_Bias_" + pName);
            out3 = new AddColWise<DTYPE>(out1, pBias_xh, "_Add_Bxh" + pName);

            Tensorholder<DTYPE> *pBias_hh = new Tensorholder<DTYPE>(Tensor<DTYPE>::Constants(1, 1, 1, 1, pNumOutputCol, 0.f), "Add_Bias_" + pName);
            out4 = new AddColWise<DTYPE>(out2, pBias_hh, "_Add_Bhh" + pName);
        }

        out5 = new AddColWise<DTYPE>(out3, out4, "_Add_ " + pName);

        hidden_next = new Tanh<float>(out5, "Tanh");

        return TRUE;
    }
};

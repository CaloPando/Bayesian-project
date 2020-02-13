import pystan
import pickle

#These file should be run before using the class Neal_NN as it initializes the models
#Used with Stan and pikels than to a repository so that they can be unpikeled when
#the class is initialized thus saving time

directory=r'C:\Users\Franz Liszt\Miniconda3\envs\Bayesian_project\Lib\site-packages'

model = """
data{
    int<lower=0> normalized;
    int<lower=0> L; //number of layers
    int<lower=0> H; //number of hidden nodes
    int<lower=0> D; //size data
    int<lower=0> I; //size input
    int<lower=0> O; //size output
    matrix[O,D] y; //target response
    matrix[I,D] x; //covariates 

    //parameters for hyperparameters
    matrix[H,H] m_w[L-1];
    matrix[H,I] m_w_i;
    matrix[O,H] m_w_o;
    matrix[H,L] m_b;
    vector[O] m_b_o;
    vector[L] m_t;
    real m_t_i;

    real<lower=0> alpha;
    real<lower=0> beta;


    }
parameters{
    matrix[H,H] w[L-1]; //weights between hidden layers
    matrix[H,I] w_I; //input weights
    matrix[O,H] w_O; //output weights
    matrix[H,L] b; //biases
    vector[L] t; //offset
    vector[O] b_O; //output biases 
    vector[I] t_I; //input offset   

    //hyperparameters
    real<lower=0> s_t; //Common sigma for offset of hidden layers
    real<lower=0> s_t_I; //Common sigma for offset of input units
    vector<lower=0>[L] s_b; //Common sigma for biases of hidden layers
    real<lower=0> s_b_O; //Common sigma for biases of output units
    vector<lower=0>[L] s_w; //Common sigma for weights of hidden layers
    real<lower=0> s_I; //Common sigma for weights of input units
    real<lower=0> s_O;

    }

transformed parameters{
    vector[H] v_h[D]; //output hidden layers
    vector[O] v_O[D]; //output

    //input pass
    for(d in 1:D){   
        v_h[d]=tanh(w_I*(x[:,d]+t_I)+b[:,1]);
    }
    //hidden passes
    for(d in 1:D){  
        for(l in 1:L-1){
            v_h[d]=tanh(w[l]*(v_h[d]+t[l])+b[:,l+1]);
        }
    }
    //output pass

    for(d in 1:D){ 
        if(normalized){  
            v_O[d]=tanh(w_O*(v_h[d]+t[L])+b_O);
        }
        else{
            v_O[d]=w_O*(v_h[d]+t[L])+b_O;
        }
    }


    }

model{
    //output
    for(d in 1:D){
       y[:,d]~normal(v_O[d],sqrt(s_O));
    }

    //weights
    for(l in 1:L-1){
        for(h1 in 1:H){
            for(h2 in 1:H){
            w[l][h2,h1]~normal(m_w[l,h1,h2],sqrt(s_w[l]));
            }
        }
    }
    //output weights
    for(h in 1:H){
        for(o in 1:O){
          w_O[o,h]~normal(m_w_o[o,h],sqrt(s_w[L]));     
        }
    }

    //input weights
    for(h in 1:H){
        for(i in 1:I){
          w_I[h,i]~normal(m_w_i[h,i],sqrt(s_I));     
        }
    }

    //biases
    for(l in 1:L){
        for(h in 1:H){
          b[h,l]~normal(m_b[h,l],sqrt(s_b[l]));      
        }
    }

    //output biases
    for(o in 1:O){
          b_O[o]~normal(m_b_o[o],sqrt(s_b_O));      
    }

    //offsets
    for(l in 1:L){
        t[l]~normal(m_t[l],sqrt(s_t));
    }

    t_I~normal(m_t_i,sqrt(s_t_I));


    //hyperparameters (all sigmas have inverse gamma distribution)
    s_t~inv_gamma(alpha,beta);
    s_t_I~inv_gamma(alpha,beta);

    for(l in 1:L){
        s_b~inv_gamma(alpha,beta);
    }

    s_b_O~inv_gamma(alpha,beta);

    for(l in 1:L){
        s_w[l]~inv_gamma(alpha,beta);
    }

    s_I~inv_gamma(alpha,beta);

    s_O~inv_gamma(alpha,beta);
    }
    

generated quantities 
{
  	vector[D] log_lik;
  	for (d in 1:D){
    		log_lik[d] = normal_lpdf(y[:,d] | v_O[d],s_O);
  	}
}


"""


model_uncentered = """
data{
    int<lower=0> normalized;
    int<lower=0> L; //number of layers
    int<lower=0> H; //number of hidden nodes
    int<lower=0> D; //size data
    int<lower=0> I; //size input
    int<lower=0> O; //size output
    matrix[O,D] y; //target response
    matrix[I,D] x; //covariates 

    //parameters for hyperparameters
    matrix[H,H] m_w[L-1];
    matrix[H,I] m_w_i;
    matrix[O,H] m_w_o;
    matrix[H,L] m_b;
    vector[O] m_b_o;
    vector[L] m_t;
    real m_t_i;
    real<lower=0> alpha;
    real<lower=0> beta;
    }
parameters{
    matrix[H,H] w_tilde[L-1]; //weights between hidden layers
    matrix[H,I] w_I_tilde; //input weights
    matrix[O,H] w_O_tilde; //output weights
    matrix[H,L] b_tilde; //biases
    vector[L] t_tilde; //offset
    vector[O] b_O_tilde; //output biases 
    vector[I] t_I_tilde; //input offset   

    //hyperparameters
    real<lower=0> s_t; //Common sigma for offset of hidden layers
    real<lower=0> s_t_I; //Common sigma for offset of input units
    vector<lower=0>[L] s_b; //Common sigma for biases of hidden layers
    real<lower=0> s_b_O; //Common sigma for biases of output units
    vector<lower=0>[L] s_w; //Common sigma for weights of hidden layers
    real<lower=0> s_I; //Common sigma for weights of input units


    real<lower=0> s_O;

    }

transformed parameters{
    vector[H] v_h[D]; //output hidden layers
    vector[O] v_O[D]; //output
    matrix[H,H] w[L-1]; //weights between hidden layers
    matrix[H,I] w_I; //input weights
    matrix[O,H] w_O; //output weights
    matrix[H,L] b; //biases
    vector[L] t; //offset
    vector[O] b_O; //output biases 
    vector[I] t_I; //input offset   

    //weights
    for(l in 1:L-1){
        for(h1 in 1:H){
            for(h2 in 1:H){
            w[l][h2,h1]=w_tilde[l][h2,h1]*s_w[l]+m_w[l][h1,h2];
            }
        }
    }
    //output weights
    for(h in 1:H){
        for(o in 1:O){
          w_O[o,h]=w_O_tilde[o,h]*sqrt(s_w[L])+m_w_o[o,h];     
        }
    }

    //input weights
    for(h in 1:H){
        for(i in 1:I){
          w_I[h,i]=w_I_tilde[h,i]*sqrt(s_I)+m_w_i[h,i];     
        }
    }

    //biases
    for(l in 1:L){
        for(h in 1:H){
          b[h,l]=b_tilde[h,l]*sqrt(s_b[l])+m_b[h,l];      
        }
    }

    //output biases
    for(o in 1:O){
          b_O[o]=b_O_tilde[o]*sqrt(s_b_O)+m_b_o[o];      
    }

    //offsets
    for(l in 1:L){
        t[l]=t_tilde[l]*sqrt(s_t)+m_t[l];
    }

    t_I=t_I_tilde*sqrt(s_t_I)+m_t_i;




    //input pass
    for(d in 1:D){   
        v_h[d]=tanh(w_I*(x[:,d]+t_I)+b[:,1]);
    }
    //hidden passes
    for(d in 1:D){  
        for(l in 1:L-1){
            v_h[d]=tanh(w[l]*(v_h[d]+t[l])+b[:,l+1]);
        }
    }
    //output pass

    for(d in 1:D){ 
        if(normalized){  
            v_O[d]=tanh(w_O*(v_h[d]+t[L])+b_O);
        }
        else{
            v_O[d]=w_O*(v_h[d]+t[L])+b_O;
        }
    }





    }

model{
    //output
    for(d in 1:D){
       y[:,d]~normal(v_O[d],sqrt(s_O));
    }

    //weights
    for(l in 1:L-1){
        for(h1 in 1:H){
            for(h2 in 1:H){
            w_tilde[l][h2,h1]~normal(0,1);
            }
        }
    }
    //output weights
    for(h in 1:H){
        for(o in 1:O){
          w_O_tilde[o,h]~normal(0,1);     
        }
    }

    //input weights
    for(h in 1:H){
        for(i in 1:I){
          w_I_tilde[h,i]~normal(0,1);     
        }
    }

    //biases
    for(l in 1:L){
        for(h in 1:H){
          b_tilde[h,l]~normal(0,1);      
        }
    }

    //output biases
    for(o in 1:O){
          b_O_tilde[o]~normal(0,1);      
    }

    //offsets
    for(l in 1:L){
        t_tilde[l]~normal(0,1);
    }

    t_I_tilde~normal(0,1);


    //hyperparameters (all sigmas have inverse gamma distribution)
    s_t~inv_gamma(alpha,beta);
    s_t_I~inv_gamma(alpha,beta);

    for(l in 1:L){
        s_b~inv_gamma(alpha,beta);
    }

    s_b_O~inv_gamma(alpha,beta);

    for(l in 1:L){
        s_w[l]~inv_gamma(alpha,beta);
    }

    s_I~inv_gamma(alpha,beta);

    s_O~inv_gamma(alpha,beta);
    }


generated quantities 
{
  	vector[D] log_lik;
  	for (d in 1:D){
    		log_lik[d] = normal_lpdf(y[:,d] | v_O[d],s_O);
  	}
}

"""










# run these lines once to create the models
sm = pystan.StanModel(model_code=model)
with open(directory+'model.pkl', 'wb') as f:
    pickle.dump(sm, f, protocol=pickle.HIGHEST_PROTOCOL)

sm = pystan.StanModel(model_code=model_uncentered)
with open(directory+'model.pkl', 'wb') as f:
    pickle.dump(sm, f, protocol=pickle.HIGHEST_PROTOCOL)





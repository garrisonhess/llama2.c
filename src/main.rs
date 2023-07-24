use ndarray::s;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Array3;
use ndarray::ArrayView1;
use ndarray::ArrayViewMut1;
use serde::{Deserialize, Serialize};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::mem;
use std::slice;

#[derive(Debug, Serialize, Deserialize)]
struct Config {
    dim: i32,
    hidden_dim: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
}

#[derive(Debug, Clone)]
struct TransformerWeights {
    token_embedding_table: Array2<f32>, // (vocab_size, dim)
    rms_att_weight: Array2<f32>,        // (layer, dim) rmsnorm weights
    rms_ffn_weight: Array2<f32>,        // (layer, dim)
    wq: Array3<f32>,                    // (layer, dim, dim)
    wk: Array3<f32>,                    // (layer, dim, dim)
    wv: Array3<f32>,                    // (layer, dim, dim)
    wo: Array3<f32>,                    // (layer, dim, dim)
    w1: Array3<f32>,                    // (layer, hidden_dim, dim)
    w2: Array3<f32>,                    // (layer, dim, hidden_dim)
    w3: Array3<f32>,                    // (layer, hidden_dim, dim)
    rms_final_weight: Array1<f32>,      // (dim,)
    freq_cis_real: Array2<f32>,         // (seq_len, dim/2)
    freq_cis_imag: Array2<f32>,         // (seq_len, dim/2)
}

#[derive(Debug, Clone)]
struct TransformerWeightsOld {
    token_embedding_table: Vec<f32>, // (vocab_size, dim)
    rms_att_weight: Vec<f32>,        // (layer, dim) rmsnorm weights
    rms_ffn_weight: Vec<f32>,        // (layer, dim)
    wq: Vec<f32>,                    // (layer, dim, dim)
    wk: Vec<f32>,                    // (layer, dim, dim)
    wv: Vec<f32>,                    // (layer, dim, dim)
    wo: Vec<f32>,                    // (layer, dim, dim)
    w1: Vec<f32>,                    // (layer, hidden_dim, dim)
    w2: Vec<f32>,                    // (layer, dim, hidden_dim)
    w3: Vec<f32>,                    // (layer, hidden_dim, dim)
    rms_final_weight: Vec<f32>,      // (dim,)
    freq_cis_real: Vec<f32>,         // (seq_len, dim/2)
    freq_cis_imag: Vec<f32>,         // (seq_len, dim/2)
}

impl TransformerWeights {
    fn try_new(config: &Config, file: &mut File) -> Result<Self, Box<dyn Error>> {
        let mut weights = TransformerWeightsOld {
            token_embedding_table: vec![0.0; (config.vocab_size * config.dim) as usize],
            rms_att_weight: vec![0.0; (config.n_layers * config.dim) as usize],
            rms_ffn_weight: vec![0.0; (config.n_layers * config.dim) as usize],
            wq: vec![0.0; (config.n_layers * config.dim * config.dim) as usize],
            wk: vec![0.0; (config.n_layers * config.dim * config.dim) as usize],
            wv: vec![0.0; (config.n_layers * config.dim * config.dim) as usize],
            wo: vec![0.0; (config.n_layers * config.dim * config.dim) as usize],
            w1: vec![0.0; (config.n_layers * config.hidden_dim * config.dim) as usize],
            w2: vec![0.0; (config.n_layers * config.dim * config.hidden_dim) as usize],
            w3: vec![0.0; (config.n_layers * config.hidden_dim * config.dim) as usize],
            rms_final_weight: vec![0.0; config.dim as usize],
            freq_cis_real: vec![0.0; (config.seq_len * config.dim / 2) as usize],
            freq_cis_imag: vec![0.0; (config.seq_len * config.dim / 2) as usize],
        };

        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.token_embedding_table.as_mut_ptr() as *mut u8,
                weights.token_embedding_table.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.rms_att_weight.as_mut_ptr() as *mut u8,
                weights.rms_att_weight.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.wq.as_mut_ptr() as *mut u8,
                weights.wq.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.wk.as_mut_ptr() as *mut u8,
                weights.wk.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.wv.as_mut_ptr() as *mut u8,
                weights.wv.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.wo.as_mut_ptr() as *mut u8,
                weights.wo.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.rms_ffn_weight.as_mut_ptr() as *mut u8,
                weights.rms_ffn_weight.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.w1.as_mut_ptr() as *mut u8,
                weights.w1.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.w2.as_mut_ptr() as *mut u8,
                weights.w2.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.w3.as_mut_ptr() as *mut u8,
                weights.w3.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.rms_final_weight.as_mut_ptr() as *mut u8,
                weights.rms_final_weight.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        let head_size = (config.dim / config.n_heads) as usize;
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.freq_cis_real.as_mut_ptr() as *mut u8,
                config.seq_len as usize * head_size / 2 * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.freq_cis_imag.as_mut_ptr() as *mut u8,
                config.seq_len as usize * head_size / 2 * mem::size_of::<f32>(),
            )
        })
        .unwrap();

        let n_layers = config.n_layers as usize;
        let dim = config.dim as usize;
        let hidden_dim = config.hidden_dim as usize;
        let seq_len = config.seq_len as usize;
        let vocab_size = config.vocab_size as usize;

        let token_embedding_table =
            Array2::from_shape_vec((vocab_size, dim), weights.token_embedding_table)?;
        let rms_att_weight = Array2::from_shape_vec((n_layers, dim), weights.rms_att_weight)?;
        let rms_ffn_weight = Array2::from_shape_vec((n_layers, dim), weights.rms_ffn_weight)?;
        let wq = Array3::from_shape_vec((n_layers, dim, dim), weights.wq)?;
        let wk = Array3::from_shape_vec((n_layers, dim, dim), weights.wk)?;
        let wv = Array3::from_shape_vec((n_layers, dim, dim), weights.wv)?;
        let wo = Array3::from_shape_vec((n_layers, dim, dim), weights.wo)?;
        let w1 = Array3::from_shape_vec((n_layers, hidden_dim, dim), weights.w1)?;
        let w2 = Array3::from_shape_vec((n_layers, dim, hidden_dim), weights.w2)?;
        let w3 = Array3::from_shape_vec((n_layers, hidden_dim, dim), weights.w3)?;
        let rms_final_weight = Array1::from_shape_vec((dim,), weights.rms_final_weight)?;
        let freq_cis_real = Array2::from_shape_vec((seq_len, dim / 2), weights.freq_cis_real)?;
        let freq_cis_imag = Array2::from_shape_vec((seq_len, dim / 2), weights.freq_cis_imag)?;

        let weights_arr = Self {
            token_embedding_table,
            rms_att_weight,
            rms_ffn_weight,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            rms_final_weight,
            freq_cis_real,
            freq_cis_imag,
        };

        Ok(weights_arr)
    }
}

#[derive(Debug, Clone)]
struct RunState {
    x: Array1<f32>,           // activation at current time stamp (dim,)
    xb: Array1<f32>,          // same, but inside a residual branch (dim,)
    xb2: Array1<f32>,         // an additional buffer just for convenience (dim,)
    hb: Array1<f32>,          // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Array1<f32>,         // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Array1<f32>,           // query (dim,)
    k: Array1<f32>,           // key (dim,)
    v: Array1<f32>,           // value (dim,)
    att: Array1<f32>,         // buffer for scores/attention values (seq_len,)
    logits: Array1<f32>,      // output logits
    key_cache: Array3<f32>,   // (layer, seq_len, dim)
    value_cache: Array3<f32>, // (layer, seq_len, dim)
}

impl RunState {
    fn new(config: &Config) -> RunState {
        RunState {
            x: Array1::zeros((config.dim as usize,)),
            xb: Array1::zeros((config.dim as usize,)),
            xb2: Array1::zeros((config.dim as usize,)),
            hb: Array1::zeros((config.hidden_dim as usize,)),
            hb2: Array1::zeros((config.hidden_dim as usize,)),
            q: Array1::zeros((config.dim as usize,)),
            k: Array1::zeros((config.dim as usize,)),
            v: Array1::zeros((config.dim as usize,)),
            att: Array1::zeros((config.seq_len as usize,)),
            logits: Array1::zeros((config.vocab_size as usize,)),
            key_cache: Array3::zeros((
                config.n_layers as usize,
                config.seq_len as usize,
                config.dim as usize,
            )),
            value_cache: Array3::zeros((
                config.n_layers as usize,
                config.seq_len as usize,
                config.dim as usize,
            )),
        }
    }
}

fn rmsnorm(mut o: ArrayViewMut1<f32>, x: ArrayView1<f32>, weight: ArrayView1<f32>) {
    // calculate sum of squares
    let mut ss: f32 = x.mapv(|x| x.powi(2)).sum();
    ss /= o.len() as f32;
    ss += 1e-5_f32;
    ss = 1.0 / ss.sqrt();

    // normalize and scale
    o.assign(&(ss * &x * &weight));
}

fn sample(probabilities: ArrayView1<f32>) -> usize {
    // sample index from probabilities, they must sum to 1
    let r: f32 = rand::random();
    let mut cdf = 0.0;
    for (i, &prob) in probabilities.iter().enumerate() {
        cdf += prob;
        if r < cdf {
            return i;
        }
    }
    probabilities.len() - 1 // in case of rounding errors
}

fn softmax(mut x: ArrayViewMut1<f32>) {
    if x.len() == 1 {
        x[0] = 1.0;
        return;
    }

    // find max value (for numerical stability)
    let mut max_val = x[0];
    for &val in x.slice(s![1..]) {
        if val > max_val {
            max_val = val;
        }
    }

    // e^x
    for val in x.iter_mut() {
        *val = (*val - max_val).exp();
    }

    // normalize
    x /= x.sum();
}

// Matmuls are W (d,n) @ x (n,) -> xout (d,)
fn transformer(
    token: usize,
    pos: usize,
    config: &Config,
    state: &mut RunState,
    weights: &TransformerWeights,
) {
    // a few convenience variables
    let dim = config.dim as usize;
    let head_size = (dim / config.n_heads as usize) as usize;

    // copy the token embedding into x
    state
        .x
        .assign(&weights.token_embedding_table.slice(s![token, ..]));

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    let freq_cis_real_row = weights.freq_cis_real.slice(s![pos, ..head_size / 2]);
    let freq_cis_imag_row = weights.freq_cis_imag.slice(s![pos, ..head_size / 2]);

    // forward all the layers
    for l in 0..config.n_layers as usize {
        // attention rmsnorm
        rmsnorm(
            (&mut state.xb).into(),
            (&state.x).into(),
            weights.rms_att_weight.slice(s![l, ..]),
        );

        // qkv matmuls for this position
        state
            .q
            .assign(&weights.wq.slice(s![l, .., ..]).dot(&state.xb));
        state
            .k
            .assign(&weights.wk.slice(s![l, .., ..]).dot(&state.xb));
        state
            .v
            .assign(&weights.wv.slice(s![l, .., ..]).dot(&state.xb));

        // apply RoPE rotation to the q and k vectors for each head
        for h in 0..config.n_heads as usize {
            // get the q and k vectors for this head
            let mut q = state.q.slice_mut(s![h * head_size..(h + 1) * head_size]);
            let mut k = state.k.slice_mut(s![h * head_size..(h + 1) * head_size]);

            // rotate q and k by the freq_cis_real and freq_cis_imag
            for i in (0..head_size).step_by(2) {
                let q0 = q[i];
                let q1 = q[i + 1];
                let k0 = k[i];
                let k1 = k[i + 1];
                let fcr = freq_cis_real_row[i / 2];
                let fci = freq_cis_imag_row[i / 2];
                q[i] = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;
                k[i] = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        // let loff = l * config.seq_len as usize * dim; // kv cache layer offset for convenience
        state.key_cache.slice_mut(s![l, pos, ..]).assign(&state.k);
        state.value_cache.slice_mut(s![l, pos, ..]).assign(&state.v);

        // multihead attention. iterate over all heads
        for h in 0..config.n_heads as usize {
            // get the query vector for this head
            let q = state.q.slice(s![h * head_size..(h + 1) * head_size]);

            // iterate over all timesteps, including the current one
            for t in 0..=pos {
                // get the key vector for this head and at this timestep
                let k = state
                    .key_cache
                    .slice(s![l, t, h * head_size..(h + 1) * head_size]);

                // calculate the attention score as the scaled dot product of q and k
                let score = k.dot(&q) / (head_size as f32).sqrt();

                // save the score to the attention buffer
                state.att[t] = score;
            }

            softmax(state.att.slice_mut(s![..=pos]));

            // weighted sum of the values, store back into xb
            for i in 0..head_size {
                let mut val = 0.0;
                for t in 0..=pos {
                    // note bad locality
                    val += state.att[t] * state.value_cache[[l, t, h * head_size + i]];
                }
                state.xb[h * head_size + i] = val;
            }
        }

        // final matmul to get the output of the attention
        state
            .xb2
            .assign(&weights.wo.slice(s![l, .., ..]).dot(&state.xb));

        // residual connection back into x
        state.x += &state.xb2;

        // ffn rmsnorm
        rmsnorm(
            (&mut state.xb).into(),
            (&state.x).into(),
            weights.rms_ffn_weight.slice(s![l, ..]),
        );

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        state
            .hb
            .assign(&weights.w1.slice(s![l, .., ..]).dot(&state.xb));
        state
            .hb2
            .assign(&weights.w3.slice(s![l, .., ..]).dot(&state.xb));

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for val in &mut state.hb {
            *val *= 1.0 / (1.0 + (-*val).exp());
        }

        // elementwise multiply with w3(x)
        state.hb *= &state.hb2;

        // final matmul to get the output of the ffn
        state
            .xb
            .assign(&weights.w2.slice(s![l, .., ..]).dot(&state.hb));

        // residual connection
        state.x += &state.xb;
    }

    // final rmsnorm
    let temp_x = state.x.clone();
    rmsnorm(
        (&mut state.x).into(),
        (&temp_x).into(),
        (&weights.rms_final_weight).into(),
    );

    // classifier into logits
    state
        .logits
        .assign(&weights.token_embedding_table.dot(&state.x));
}

fn argmax(v: ArrayView1<f32>) -> usize {
    let mut max_i = 0;
    let mut max_p = v[0];
    for (i, &val) in v.iter().enumerate().skip(1) {
        if val > max_p {
            max_i = i;
            max_p = val;
        }
    }
    max_i
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let checkpoint = args
        .next()
        .expect("Usage: llama2-rust <checkpoint_file> [temperature]");
    let temperature = args
        .next()
        .map(|t| t.parse().expect("Invalid temperature"))
        .unwrap_or(0.9_f32);

    let mut file = File::open(&checkpoint).expect("Failed to open checkpoint file");
    let config: Config = bincode::deserialize_from(&mut file).expect("Failed to read config");
    let mut state = RunState::new(&config);
    let weights = TransformerWeights::try_new(&config, &mut file)?;
    let mut token = 1; // 1 = BOS token in Llama-2 sentencepiece
    dbg!(&config);

    for pos in 0..config.seq_len as usize {
        transformer(token, pos, &config, &mut state, &weights);

        // advance
        token = if temperature == 0.0_f32 {
            // greedy argmax sampling
            argmax((&state.logits).into())
        } else {
            // apply the temperature to the logits
            for q in 0..config.vocab_size as usize {
                state.logits[q] /= temperature;
            }

            // apply softmax to the logits to get the probabilities for next token
            softmax((&mut state.logits).into());

            // we now want to sample from this distribution to get the next token
            sample((&state.logits).into())
        };

        println!("{token}");
    }

    Ok(())
}

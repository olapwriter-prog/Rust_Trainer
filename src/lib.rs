//! On-device LoRA trainer for the agentOS Qwen engine.
//!
//! See `README.md` for the design. Module map:
//! * [`frozen`], [`linear`], [`ops`], [`attention`], [`deltanet`], [`model`]: forward + backward;
//! * [`optim`], [`train`]: AdamW, schedule, SFT and DPO loops with resumable checkpoints;
//! * `rie` (feature `rie`, default): GGUF models through the Rust Inference Engine,
//!   training on its own kernels (bit-exact forward), engine-loadable GGUF adapters;
//! * [`data`], [`tokenizer`]: confirmed-only training sets from `agentd ctl learn`;
//! * [`safetensors`], [`hf`], [`adapter`]: model loading and the adapter format;
//! * [`eval`], [`registry`], [`scheduler`]: eval gate, swap/rollback, ledger and power-aware runs.

// Same lint policy as rie-core: index loops are clearer for the numeric kernels.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

pub mod adapter;
pub mod attention;
pub mod data;
pub mod deltanet;
pub mod eval;
pub mod frozen;
pub mod hf;
pub mod linear;
pub mod model;
pub mod ops;
pub mod optim;
pub mod registry;
#[cfg(feature = "rie")]
pub mod rie;
pub mod safetensors;
pub mod scheduler;
pub mod tensor;
pub mod tokenizer;
pub mod train;

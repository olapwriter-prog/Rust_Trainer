//! On-device LoRA trainer for the agentOS Qwen engine.
//!
//! See `README.md` for the design. Module map:
//! * [`frozen`], [`linear`], [`ops`], [`attention`], [`deltanet`], [`model`]: forward + backward;
//! * [`optim`], [`train`], [`dpo`]: optimisation;
//! * [`data`], [`tokenizer`]: confirmed-only training sets from `agentd ctl learn`;
//! * [`safetensors`], [`hf`], [`adapter`]: model loading and the adapter format;
//! * [`eval`], [`registry`], [`scheduler`]: eval gate, swap/rollback, ledger and power-aware runs.

pub mod attention;
pub mod deltanet;
pub mod frozen;
pub mod linear;
pub mod model;
pub mod ops;
pub mod tensor;
pub mod adapter;
pub mod data;
pub mod eval;
pub mod hf;
pub mod optim;
pub mod registry;
pub mod safetensors;
pub mod scheduler;
pub mod tokenizer;
pub mod train;

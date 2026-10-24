#![forbid(unsafe_code)]

pub mod config;
pub mod data;
pub mod model;
pub mod optim;
pub mod runner;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

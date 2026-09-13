# ⚡ Low-Power Dataflow-Optimized Datapath for Mixed Neural Acceleration

> 🚀 A configurable hardware AI accelerator for efficient **CNN and RNN neural network inference**, featuring **multi-core MAC parallelism, INT8/INT16 quantization, runtime configuration, and hardware-software co-design**.

---

## 🧠 Project Overview

Deep Neural Networks (DNNs) perform a large number of **Multiply-Accumulate (MAC)** operations during inference, creating significant challenges for latency, power consumption, and computational efficiency on resource-constrained edge devices.

This project presents a **configurable SystemVerilog-based neural accelerator** designed to support both:

- 🖼️ **CNN — Parallel spatial processing**
- 🔄 **RNN — Sequential temporal processing**

The accelerator combines a scalable **4-core / 8-core MAC architecture** with configurable **INT8 / INT16 precision**, enabling exploration of the trade-off between **performance, accuracy, computational resources, and power efficiency**.

---

## ✨ Key Features

- ⚡ Hardware-accelerated neural network inference
- 🧠 Unified CNN + RNN execution architecture
- 🔢 Configurable **INT8 / INT16 quantization**
- 🔲 Scalable **4-core / 8-core MAC array**
- 🔀 Parallel MAC computation
- 🎛️ Runtime-configurable execution modes
- 💾 Accumulator-based datapath
- 🧩 SystemVerilog RTL implementation
- 🐍 Python/PyTorch integration
- 🔗 Verilator-based hardware-software co-simulation
- 📊 Accuracy, MSE and cycle-level performance evaluation
- 🖥️ Linux environment using WSL 2
- 🌱 Designed for resource-constrained and edge AI applications

---

## 🏗️ System Architecture

```text
                 ┌──────────────────────────┐
                 │      Python / PyTorch    │
                 │  Model Training & Input  │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │      Model Parser        │
                 │  CNN / RNN Parameters    │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │       Quantization       │
                 │       INT8 / INT16       │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   Accelerator Interface  │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │   Verilator C++ Wrapper  │
                 └────────────┬─────────────┘
                              │
                              ▼
       ┌─────────────────────────────────────────────┐
       │        ⚙️ SystemVerilog AI Accelerator      │
       │                                             │
       │  ┌────────────┐    ┌────────────────────┐  │
       │  │ Controller │───▶│   Compute Cluster  │  │
       │  └────────────┘    └─────────┬──────────┘  │
       │                               │             │
       │                 ┌─────────────▼──────────┐  │
       │                 │ Parallel MAC Array     │  │
       │                 │      4-Core / 8-Core   │  │
       │                 └─────────────┬──────────┘  │
       │                               │             │
       │                 ┌─────────────▼──────────┐  │
       │                 │      Accumulator       │  │
       │                 └─────────────┬──────────┘  │
       └───────────────────────────────┼─────────────┘
                                       │
                                       ▼
                         ┌────────────────────────┐
                         │  Performance Evaluation │
                         │ Accuracy • MSE • Cycles │
                         └────────────────────────┘

🔬 Supported Workloads
🖼️ CNN Inference

The CNN evaluation uses the MNIST dataset with 28×28 grayscale images.

Input Image
    ↓
Conv2D
    ↓
ReLU + MaxPool
    ↓
Conv2D
    ↓
ReLU + MaxPool
    ↓
Flatten
    ↓
FC Layer 1 ───────► ⚡ Hardware Accelerated
    ↓
ReLU
    ↓
FC Layer 2
    ↓
🎯 Predicted Digit

The first fully connected layer performs a 784 × 128 matrix-vector multiplication, making it well suited for MAC-array acceleration.

🔄 RNN Inference

The RNN evaluation uses a simplified LSTM-inspired recurrent formulation for sine-wave prediction.

Input Sequence
      ↓
Sequence Window
      ↓
RNN / LSTM-Inspired Model
      ↓
Hidden State
      ↓
Matrix-Vector Multiplication
      ↓
Tanh Activation
      ↓
📈 Predicted Output
      ↓
MSE Evaluation

⚠️ The current hardware implementation uses a simplified recurrent formulation and does not implement the complete LSTM gate equations.

🔢 Quantization

The accelerator supports two configurable precision modes:

Mode	Operand Width	Accumulator	Purpose
🔹 INT8	8-bit	16-bit	Lower computational cost
🔸 INT16	16-bit	32-bit	Higher numerical precision
INT8
quantized_value = round(float_value × 16)
INT16
quantized_value = round(float_value × 256)

INT16 provides improved numerical precision at the cost of wider datapaths and accumulators.

⚙️ Hardware Configurations

Four configurations were evaluated:

┌───────────────────────┐
│  4-Core INT8          │
├───────────────────────┤
│  4-Core INT16         │
├───────────────────────┤
│  8-Core INT8          │
├───────────────────────┤
│  8-Core INT16         │
└───────────────────────┘

This enables evaluation of:

⚡ Parallelism
🔢 Numerical precision
🧮 Computational latency
📊 Prediction accuracy
💾 Hardware resource utilization
📊 Performance Results
🖼️ CNN
Configuration	FC1 Cycles
4-Core	100,352
8-Core	50,176

🚀 Speedup: 2.0×

Doubling the number of MAC cores from 4 to 8 reduced the cycle count by approximately 50%.

🔄 RNN
Configuration	MSE	Cycles
4-Core INT8	1.354	40,960
4-Core INT16	1.241	40,960
8-Core INT8	1.354	20,480
8-Core INT16	1.241	20,480
🏆 Key Results
⚡ 2.0× speedup with 8-core execution
📉 50% cycle reduction from 4-core to 8-core operation
🎯 INT16 achieved lower RNN MSE
🔢 INT16 produced approximately 8.3% lower error than INT8 for the evaluated RNN task
🔄 Parallelism improved performance without changing RNN accuracy
🧩 Configurable precision enables accuracy-performance trade-offs
🛠️ Technology Stack
💻 Programming & AI
🐍 Python
🔥 PyTorch
🔢 NumPy
📊 Matplotlib
⚙️ Hardware Design
🧩 SystemVerilog
🔲 RTL Design
⚡ MAC Architecture
🧮 Fixed-Point Arithmetic
🔢 INT8 / INT16 Quantization
🏗️ Multi-Core Parallel Architecture
🎛️ Configurable Datapath
🔗 Hardware-Software Integration
🔵 Verilator
💻 C++
🐧 WSL 2
🛠️ GCC
🔄 Hardware-Software Co-Design
🔄 Hardware-Software Co-Design Flow
PyTorch Model
      ↓
Model Preparation
      ↓
Quantization
      ↓
Python Accelerator Interface
      ↓
C++ Verilator Wrapper
      ↓
SystemVerilog RTL
      ↓
MAC Array
      ↓
Hardware Computation
      ↓
Performance Evaluation
      ↓
Accuracy / MSE / Cycle Analysis
📁 Project Structure
📦 mixed-neural-accelerator
│
├── 📁 rtl/
│   ├── mac_unit.sv
│   ├── compute_cluster.sv
│   ├── controller.sv
│   └── accelerator_top.sv
│
├── 📁 python/
│   ├── cnn_model.py
│   ├── rnn_model.py
│   ├── quantization.py
│   └── accelerator_interface.py
│
├── 📁 verilator/
│   └── verilator_wrapper.cpp
│
├── 📁 simulation/
│   ├── testbench.sv
│   └── test_data/
│
├── 📁 results/
│   ├── cnn_results/
│   └── rnn_results/
│
├── 📁 documentation/
│   ├── paper.pdf
│   └── poster.pdf
│
└── 📄 README.md

📌 Modify the folder names above according to the actual files in the repository.

🎯 Project Objectives
🧠 Design a unified accelerator supporting CNN and RNN workloads
⚡ Improve neural inference performance using hardware parallelism
🔢 Investigate INT8 vs INT16 precision trade-offs
🔲 Implement scalable 4-core and 8-core MAC architectures
🎛️ Enable configurable hardware execution
🔗 Integrate Python/PyTorch models with SystemVerilog RTL
📊 Evaluate accuracy, MSE and cycle-level performance
🌱 Explore efficient AI acceleration for resource-constrained systems
💡 Why Hardware Acceleration?

Software-only neural inference can suffer from:

⏳ High computation latency
🔄 Sequential MAC execution
💾 Inefficient memory access
⚡ Limited parallelism
🔋 Higher computational overhead

The proposed accelerator addresses these challenges using:

        Software AI
            │
            ▼
    ┌───────────────┐
    │ Quantization  │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │ Parallel MACs │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │ Accumulation  │
    └───────┬───────┘
            │
            ▼
     Faster Inference
🌍 Applications

Potential applications include:

🤖 Edge AI systems
📷 Intelligent cameras
🎥 Video analytics
🗣️ Speech processing
🧠 Neural inference engines
🚗 Autonomous systems
📡 IoT devices
🏭 Industrial AI
⚡ Low-power embedded AI
🔬 AI accelerator research
🔮 Future Improvements
🚀 Support larger MAC arrays
🧠 Implement complete LSTM gate architecture
🔢 Explore INT4 / mixed-precision inference
💾 Optimize memory hierarchy and data movement
⚡ FPGA-based hardware implementation
🔋 Detailed power and energy measurements
🎛️ Dynamic runtime precision selection
🔀 Improved multi-stream scheduling
🧩 Support additional neural network architectures
📚 Research Focus

This project focuses on:

Hardware AI Acceleration • Neural Network Inference • CNN Acceleration • RNN Acceleration • SystemVerilog RTL • Hardware-Software Co-Design • MAC Architecture • Quantization • Fixed-Point Arithmetic • Parallel Computing • Edge AI • Low-Power Computing • FPGA/ASIC Design

👨‍💻 Project Team
Gurukiran B H

⭐ Highlights
⚡ 2× CNN speedup
⚡ 2× RNN speedup
🔲 4-Core / 8-Core MAC Array
🔢 INT8 / INT16 Precision
🧠 CNN + RNN Support
🔗 Python + PyTorch + Verilator
🧩 SystemVerilog RTL
📊 Cycle-Level Performance Analysis
🌱 Edge AI & Low-Power Computing
📜 License

This project is developed for academic and research purposes.

⭐ If you find this project interesting, consider giving the repository a star!




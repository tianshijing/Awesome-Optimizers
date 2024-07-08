# Awesome-optimizer-PyTorch

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

Welcome to our repository, a meticulously curated collection of optimization algorithms implemented in PyTorch, designed to cater to the diverse needs of the machine learning research community.

## Introduction to the Repository

In the realm of machine learning, the choice of an optimizer is as crucial as the architecture of the model itself. This repository aims to provide a comprehensive exploration of various optimization algorithms, with a focus on their implementation and application in PyTorch, facilitating researchers and practitioners in making informed decisions for their projects.

### Our Latest Work: A Decade’s Battle on the Bias of Vision Backbone and Optimizer

<details>
<summary><strong>Click to Expand</strong></summary>

<h3>A Decade’s Battle on the Bias of Vision Backbone and Optimizer</h3>

Over the past decade, there has been rapid progress in vision backbones and an evolution of deep optimizers from SGD to Adam variants. Our latest paper, for the first time, investigates the relationship between vision network design and optimizer selection. We conducted comprehensive benchmarking studies on mainstream vision backbones and widely-used optimizers, revealing an intriguing phenomenon termed **Backbone-Optimizer Coupling Bias (BOCB)**.

- **Classical ConvNets** like VGG and ResNet show a strong co-dependency with SGD.
- **Modern architectures** such as ViTs and ConvNeXt demonstrate a strong coupling with optimizers with adaptive learning rates like AdamW.

More importantly, we uncover the adverse impacts of BOCB on popular backbones in real-world practice, such as additional tuning time and resource overhead, indicating remaining challenges and potential risks. Through in-depth analysis and apples-to-apples comparisons, we surprisingly observe that specific types of network architecture can significantly mitigate BOCB, which might serve as promising guidelines for future backbone design.

We hope this work can inspire the community to further question the long-held assumptions on vision backbones and optimizers, consider BOCB in future studies, and contribute to more robust, efficient, and effective vision systems. It's time to go beyond the usual choices and confront the elephant in the room. The source code and models are publicly available.

</details>

### Visualizing Performance Differences

To illustrate the performance differences of 20 optimizers across various vision backbones under optimal parameter settings, we have included the figure ![Optimizer Accuracy](Fig/acc.jpg). This figure provides a clear visual representation of how different optimizers perform in different scenarios.

Additionally, we have categorized classic optimizers into four main types, as shown in the following image:

![Optimizer Categories](Fig/optimizer.jpg)

This classification helps in understanding the underlying principles and applications of these optimizers.


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---


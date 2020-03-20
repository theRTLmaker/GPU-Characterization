# Non-Conventional AMD GPU Characterization Tool and Methodology

[![NPM Version][npm-image]][npm-url]
[![Build Status][travis-image]][travis-url]
[![Downloads Stats][npm-downloads]][npm-url]

Set of benchmarks ana analysis tools to characterize the response of AMD GPU architecture to different workloads whily using non-conventional (extreme undervoltage) DVFS settings


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites (for CentOS 7.0)


* Python 3.8
* Anaconda/Miniconda enviroment
* ROCm 3.0
* ROCblas
* MIopen-hip
* MIopen-gemm


### Installing

* Create conda enviroment for analysis tool

```
conda env create -f environment/environment.yml
```
* ROCm 3.0
```
sudo yum install rocm-dkms rock-dkms
sudo yum install rocblas
sudo yum install miopen-hip
sudo yum install miopengemm
```

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Python](https://www.python.org/) - For running and analysis script
* [C++]() - Benchmarks
* [HIP](https://github.com/ROCm-Developer-Tools/HIP) - AMD Kernels

<!-- ## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us. -->

<!-- ## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).  -->

## Authors

* **Francisco Mendes** - [TheEmbbededCoder](https://github.com/TheEmbbededCoder/)

* **IST - INESC-ID** - [INESC-ID](https://inesc-id.pt/)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* The professors Nuno Roma and Pedro Tomás for all the discussion and debate over computer architecture

[![Basic Model Interface](https://img.shields.io/badge/CSDMS-Basic%20Model%20Interface-green.svg)](https://bmi.csdms.io/)
[![Test](https://github.com/csdms/bmi-example-agentpy/actions/workflows/test.yml/badge.svg)](https://github.com/csdms/bmi-example-agentpy/actions/workflows/test.yml)

# bmi-example-agentpy

An example of using the
[Python mappings](https://github.com/csdms/bmi-python)
for the CSDMS [Basic Model Interface](https://bmi.csdms.io) (BMI)
to wrap a model written with [AgentPy](https://agentpy.readthedocs.io),
an open-source library for developing and analyzing agent-based models in Python.

## Overview

This is an example of implementing a BMI for a statistical model of diffusion
on a uniform rectangular plate.
The model, [DiffusionModel](./diffusion/diffusion_model.py),
is written with AgentPy.
In the model,
a configurable number of agents are placed at a location on the plate.
In each time step,
the agents are randomly moved zero to one grid cell,
with the probability of moving modified by a pseudo-diffusivity.
The number of agents is conserved,
so if one reaches a boundary,
it is reflected back onto the plate.

This repository is organized with the following directories:

<dl>
    <dt>diffusion</dt>
        <dd>Source for the model and a BMI implementation for the model</dd>
    <dt>examples</dt>
        <dd>Python scripts and Jupyter Notebooks that demonstrate how to run the model standalone and through its BMI</dd>
    <dt>tests</dt>
        <dd>Tests that cover the model and its BMI</dd>
</dl>

## Build/Install

This example can be built and installed on Linux, macOS, and Windows.

We recommend setting up a virtual environment--e.g., through `venv` or `conda`--to install the packages required for this example.

Use `pip` to install this example and the dependencies needed to run the sample notebooks.
```bash
pip install -e ".[examples]"
```

## Use

Try the example notebooks and scripts in the [examples](./examples/) directory. 

## Acknowledgments

This work is supported by the U.S. National Science Foundation under Award No.
[2148762](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2148762),
*Collaborative Research: Facility: CSDMS: Engaging a thriving community of practice in Earth-surface dynamics*.

The AgentPy package:

> Foramitti, J., (2021). AgentPy: A package for agent-based modeling in Python. *Journal of Open Source Software*, **6(62)**, 3065, https://doi.org/10.21105/joss.03065

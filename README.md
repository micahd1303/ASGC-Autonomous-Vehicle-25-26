# ASGC Autonomous Vehicle (2025–2026)

This repository contains the code developed for the 2025–2026 ASGC Autonomous Vehicle team.

All scripts were written and tested directly on a Raspberry Pi, then stored here for version history and organization. The project progressed from basic camera experimentation to full system integration.

## Overview

* Developed by Micah Dick
* Focused on vision-based object detection (HSV + contours)
* Integrated with an 8×8 LiDAR distance sensor
* Final system combines vision + distance sensing for autonomous control

## How to Use

* Start with the **competition scripts** — these represent the most complete system
* Use the archived/test scripts to understand development and tuning decisions
* Expect imperfections — early scripts reflect experimentation and learning

## Notes

This project was built iteratively and is not perfectly optimized. Some design choices can be improved. Reviewing earlier versions may provide insight into alternative approaches.

The `HSVThresholdBars.py` script is especially useful for tuning and visualizing HSV color thresholds and is worth reviewing for understanding color detection behavior.

When in doubt, use AI to help interpret and understand the code.

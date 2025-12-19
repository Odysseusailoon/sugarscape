I will update `scripts/run_sugarscape.py` to allow detailed configuration of the environment size, resource density, and difficulty via CLI arguments.

### 1. New CLI Arguments

I will add the following arguments to `scripts/run_sugarscape.py`:

* `--width`: Grid width (default: 50).

* `--height`: Grid height (default: 50).

* `--growback`: Resource growback rate (default: 1). Lower values (e.g., 0) make the environment "harsh/competitive".

* `--capacity`: Max resource capacity per cell (default: 4). Lower values reduce overall resource density.

### 2. Difficulty Presets

I will add a `--difficulty` argument with presets that adjust `growback` and `capacity` automatically:

* `standard`: Growback=1, Capacity=4.

* `easy`: Growback=2, Capacity=6.

* `harsh`: Growback=1, Capacity=2 (Scarce resources).

* `desert`: Growback=0, Capacity=4 (Non-renewable resources).

### 3. Variant Support

I will still include the `--variant` (`sugar` vs `spice`) argument as planned previously.

### 4. Configuration Updates

I will update the `main()` function to:

* Apply the grid dimensions (`width`, `height`).

* Apply the resource parameters (`sugar_growback_rate`, `max_sugar_capacity`, etc.) based on the arguments or difficulty preset.

* Apply the `enable_spice` setting based on `--variant`.

### Usage Example

`python3 scripts/run_sugarscape.py --mode llm --width 20 --height 20 --difficulty harsh`

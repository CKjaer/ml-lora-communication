### Generating training data

This folder, together with the ser_includes folder, includes the tools necessary to generate the plots used to train and test the various models.

To start generating setup _config.json_ per your needs. The fields the config takes, and the usual values can be seen below

| **Parameter** | **Default value** | **Possible Values** | **Description**|
| ------------------- | ---------------------------- | ----------------------------------- | ----------------------------|
| `test_id` | "" | Any string identifier | Identifier for the test or simulation. Not used for anything|
| `spreading_factor` | 7 | Integer (e.g., 7, 8, 9, etc.) | The spreading factor used in the LoRa modulation scheme.  |
| `number_of_samples` | [250] | List of integers | List of the number of samples create. Must be either a length of 1 or the same as length as _snr\_values_.|
| `snr_values` | [-6, -8, -10, -12, -14, -16] | List of integers or floats | List of Signal-to-Noise Ratio (SNR) values for the desired user. Takes only one value if _random\_dist_ is set to false. |
| `rate` | [0.0, 0.25, 0.5, 0.7, 1.0] | List of floats | List of rates used as lambda parameter for the Poisson Distribution. Not used if _random\_dist_ is set to false. |
| `plot_data` | false | Boolean (true/false) | Boolean flag to enable or disable plotting of data results. |
| `random_dist` | true | Boolean (true/false) | Boolean flag indicating if the interfering users are placed randomly. Used to toggle between SNR-based sets and SIR-based sets|
| `interf_dist` | [200, 1000, 11] | List of integers [min, max, steps] | Parameters defining the distribution of interference distances. If _random\_dist_ is set to true, _min_ and _max_ defines the distances the interfering users can take, while _steps_ is unused. If _random\_dist_ is set to false, _min_ and _max_ defines the range of SIR values, that the generating function should take, while the _steps_ define the amount of SIR values to use. |
| `line_plot` | true | Boolean (true/false) | Boolean flag to switch between line plots(**true**) and stem plots(**false**).|

In _generate\_data\_config\_test\_setup.json_ the typical setup for a test set can be seen.
In _generate\_data\_config\_training\_setup.json_ the typical setup for a training set can be seen.

Finally, the _generate\_data.sh_ file includes the necessary setup to run the system on AAU AI-LAB.
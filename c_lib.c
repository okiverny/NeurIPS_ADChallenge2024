#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AIRS_INITIAL_TIME_DIMENSION 11250
#define FGS1_INITIAL_TIME_DIMENSION 135000

/**
 * @brief Calibration pipeline for AIRS input signal
 * 
 * @param signal Input signal, with size (11250, 32, 356)
 * @param gain Gain value for Analog-to-Digital Conversion
 * @param offset Offset value for Analog-to-Digital Conversion
 * @param linear_corr Linearity correction coefficients, with size (6, 32, 356)
 * @param dark Dark current signal, with size (32, 356)
 * @param dt Time step, with size (11250)
 * @param time_binning_freq Time binning frequency
 * @param flat Flat field correction map, with size (32, 356)
 * @param hot Hot pixels map, with size (32, 356)
 * @param dead Dead pixels map, with size (32, 356)
 */
void calibration_pipeline_airs(
    double (*restrict signal)[32][356],
    double gain,
    double offset,
    double (*restrict linear_corr)[32][356],
    double (*restrict dark)[356],
    double *restrict dt,
    int time_binning_freq,
    double (*restrict flat)[356],
    bool (*restrict hot)[356],
    bool (*restrict dead)[356]
)
{
    int current_airs_time_dim = AIRS_INITIAL_TIME_DIMENSION;
    
    /* Analog-to-Digital Conversion */
    for (int j = 0; j < 32; j++)
    {
        for (int k = 0; k < 356; k++)
        {
            for (int i = 0; i < current_airs_time_dim; i++)
            {
                double temp_value = signal[i][j][k] / gain + offset;
                signal[i][j][k] = (temp_value) < 0.0 ? 0.0 : temp_value;  // Clip negative values to 0.0
            }
        }
    }
    
    /* Apply linearity correction */
    for (int j = 0; j < 32; j++)
    {
        for (int k = 0; k < 356; k++)
        {
            double temp_linear_corr[6];
            temp_linear_corr[0] = linear_corr[0][j][k];
            temp_linear_corr[1] = linear_corr[1][j][k];
            temp_linear_corr[2] = linear_corr[2][j][k];
            temp_linear_corr[3] = linear_corr[3][j][k];
            temp_linear_corr[4] = linear_corr[4][j][k];
            temp_linear_corr[5] = linear_corr[5][j][k];

            for (int i = 0; i < current_airs_time_dim; i++)
            {
                double signal_value = signal[i][j][k];
                signal[i][j][k] = (                    
                    (
                        (
                            (
                                (
                                    temp_linear_corr[5] * signal_value + temp_linear_corr[4]
                                ) * signal_value + temp_linear_corr[3]
                            ) * signal_value + temp_linear_corr[2]
                        ) * signal_value + temp_linear_corr[1]
                    ) * signal_value + temp_linear_corr[0]
                );
            }
        }
    }

    /* Dark signal */
    for (int j = 0; j < 32; j++)
    {
        for (int k = 0; k < 356; k++)
        {
            double temp_dark = dark[j][k];
            for (int i = 0; i < current_airs_time_dim; i++)
            {
                signal[i][j][k] = signal[i][j][k] - temp_dark * dt[i];
            }
        }
    }

    /**
     * Get Correlated Double Sampling 
     *
     * NOTE: The cds value is stored in the original signal array,
     *       but please note that now the time dimension is half
     *       of the original signal array.
     */
    current_airs_time_dim /= 2;
    for (int j = 0; j < 32; j++)
    {
        for (int k = 0; k < 356; k++)
        {
            for (int i = 0; i < current_airs_time_dim; i++)
            {
                signal[i][j][k] = signal[(i * 2) + 1][j][k] - signal[i * 2][j][k];
            }
        }
    }

    /**
     * Time binning
     *
     * NOTE: The time binning is stored to the original signal array,
     *       but please note that now the time dimension is different.
     */
    current_airs_time_dim /= time_binning_freq;
    for (int i = 0; i < current_airs_time_dim; i++)
    {
        double temp_signal[32][356] = {{0.0}};
        for (int j = 0; j < 32; j++)
        {
            for (int k = 0; k < 356; k++)
            {
                for (int l = 0; l < time_binning_freq; l++)
                {
                    temp_signal[j][k] += signal[(i * time_binning_freq) + l][j][k];
                }
            }
        }
        for (int j = 0; j < 32; j++)
        {
            for (int k = 0; k < 356; k++)
            {
                signal[i][j][k] = temp_signal[j][k]; 
            }
        }
    }

    /* Flat field correction */
    for (int j = 0; j < 32; j++)
    {
        for (int k = 0; k < 356; k++)
        {
            double temp_flat = flat[j][k];
            for (int i = 0; i < current_airs_time_dim; i++)
            {
                signal[i][j][k] /= temp_flat;
            }
        }
    }

    /* Dead / Hot pixels averaging */
    int check_count;
    bool checks[4];
    double (*restrict temp_signal)[32][356] = malloc(current_airs_time_dim * 32 * 356 * sizeof(double));
    for (int i = 0; i < current_airs_time_dim; i++)
    {
        for (int j = 0; j < 32; j++)
        {
            for (int k = 0; k < 356; k++)
            {
                temp_signal[i][j][k] = signal[i][j][k];
            }
        }
    }

    for (int j = 0; j < 32; j++)
    {
        for (int k = 0; k < 356; k++)
        {
            if (hot[j][k] || dead[j][k])
            {
                checks[0] = false;
                checks[1] = false;
                checks[2] = false;
                checks[3] = false;
                check_count = 0;

                if ((j > 0) && (!hot[j - 1][k]) && (!dead[j - 1][k]))
                {
                    checks[0] = true;
                    check_count++;
                }
                if ((j < 31) && (!hot[j + 1][k]) && (!dead[j + 1][k]))
                {
                    checks[1] = true;
                    check_count++;
                }
                if ((k > 0) && (!hot[j][k - 1]) && (!dead[j][k - 1]))
                {
                    checks[2] = true;
                    check_count++;
                }
                if ((k < 355) && (!hot[j][k + 1]) && (!dead[j][k + 1]))
                {
                    checks[3] = true;
                    check_count++;
                }
                
                if (check_count == 0)
                {
                    continue;
                }

                for (int i = 0; i < current_airs_time_dim; i++)
                {
                    double sum = 0.0;
                    if (checks[0])
                    {
                        sum += temp_signal[i][j - 1][k];
                    }
                    if (checks[1])
                    {
                        sum += temp_signal[i][j + 1][k];
                    }
                    if (checks[2])
                    {
                        sum += temp_signal[i][j][k - 1];
                    }
                    if (checks[3])
                    {
                        sum += temp_signal[i][j][k + 1];
                    }

                    signal[i][j][k] = sum / check_count;
                }
            }
        }
    }
    free(temp_signal); 
}

/**
 * @brief Calibration pipeline for FGS1 input signal
 * 
 * @param signal Input signal, with size (135000, 32, 32)
 * @param gain Gain value for Analog-to-Digital Conversion
 * @param offset Offset value for Analog-to-Digital Conversion
 * @param linear_corr Linearity correction coefficients, with size (6, 32, 32)
 * @param dark Dark current signal, with size (32, 32)
 * @param dt Time step, with size (135000)
 * @param time_binning_freq Time binning frequency
 * @param flat Flat field correction map, with size (32, 32)
 * @param hot Hot pixels map, with size (32, 32)
 * @param dead Dead pixels map, with size (32, 32)
 */
void calibration_pipeline_fgs1(
    double (*restrict signal)[32][32],
    double gain,
    double offset,
    double (*restrict linear_corr)[32][32],
    double (*restrict dark)[32],
    double *restrict dt,
    int time_binning_freq,
    double (*restrict flat)[32],
    bool (*restrict hot)[32],
    bool (*restrict dead)[32]
)
{
    int current_fgs1_time_dim = FGS1_INITIAL_TIME_DIMENSION;

    /* Analog-to-Digital Conversion */
    for (int j = 0; j < 32; j++)
    {
        for (int k = 0; k < 32; k++)
        {
            for (int i = 0; i < current_fgs1_time_dim; i++)
            {
                double temp_value = signal[i][j][k] / gain + offset;
                signal[i][j][k] = (temp_value) < 0.0 ? 0.0 : temp_value;  // Clip negative values to 0.0
            }
        }
    }
    
    /* Apply linearity correction */
    for (int j = 0; j < 32; j++)
    {
        for (int k = 0; k < 32; k++)
        {
            double temp_linear_corr[6];
            temp_linear_corr[0] = linear_corr[0][j][k];
            temp_linear_corr[1] = linear_corr[1][j][k];
            temp_linear_corr[2] = linear_corr[2][j][k];
            temp_linear_corr[3] = linear_corr[3][j][k];
            temp_linear_corr[4] = linear_corr[4][j][k];
            temp_linear_corr[5] = linear_corr[5][j][k];

            for (int i = 0; i < current_fgs1_time_dim; i++)
            {
                double signal_value = signal[i][j][k];
                signal[i][j][k] = (                    
                    (
                        (
                            (
                                (
                                    temp_linear_corr[5] * signal_value + temp_linear_corr[4]
                                ) * signal_value + temp_linear_corr[3]
                            ) * signal_value + temp_linear_corr[2]
                        ) * signal_value + temp_linear_corr[1]
                    ) * signal_value + temp_linear_corr[0]
                );
            }
        }
    }

    /* Dark signal */
    for (int j = 0; j < 32; j++)
    {
        for (int k = 0; k < 32; k++)
        {
            double temp_dark = dark[j][k];
            for (int i = 0; i < current_fgs1_time_dim; i++)
            {
                signal[i][j][k] = signal[i][j][k] - temp_dark * dt[i];
            }
        }
    }

    /**
     * Get Correlated Double Sampling 
     *
     * NOTE: The cds value is stored in the original signal array,
     *       but please note that now the time dimension is half
     *       of the original signal array.
     */
    current_fgs1_time_dim /= 2;
    for (int j = 0; j < 32; j++)
    {
        for (int k = 0; k < 32; k++)
        {
            for (int i = 0; i < current_fgs1_time_dim; i++)
            {
                signal[i][j][k] = signal[(i * 2) + 1][j][k] - signal[i * 2][j][k];
            }
        }
    }

    /**
     * Time binning
     *
     * NOTE: The time binning is stored to the original signal array,
     *       but please note that now the time dimension has a new size.
     */
    current_fgs1_time_dim /= time_binning_freq;
    for (int i = 0; i < current_fgs1_time_dim; i++)
    {
        double temp_signal[32][32] = {{0.0}};
        for (int j = 0; j < 32; j++)
        {
            for (int k = 0; k < 32; k++)
            {
                for (int l = 0; l < time_binning_freq; l++)
                {
                    temp_signal[j][k] += signal[(i * time_binning_freq) + l][j][k];
                }
            }
        }
        for (int j = 0; j < 32; j++)
        {
            for (int k = 0; k < 32; k++)
            {
                signal[i][j][k] = temp_signal[j][k];
            }
        }
    }

    /* Flat field correction */
    for (int j = 0; j < 32; j++)
    {
        for (int k = 0; k < 32; k++)
        {
            double temp_flat = flat[j][k];
            for (int i = 0; i < current_fgs1_time_dim; i++)
            {
                signal[i][j][k] /= temp_flat;
            }
        }
    }

    /* Dead / Hot pixels averaging */
    int check_count;
    bool checks[4];
    double (*restrict temp_signal)[32][32] = malloc(current_fgs1_time_dim * 32 * 32 * sizeof(double));
    for (int i = 0; i < current_fgs1_time_dim; i++)
    {
        for (int j = 0; j < 32; j++)
        {
            for (int k = 0; k < 32; k++)
            {
                temp_signal[i][j][k] = signal[i][j][k];
            }
        }
    }

    for (int j = 0; j < 32; j++)
    {
        for (int k = 0; k < 32; k++)
        {
            if (hot[j][k] || dead[j][k])
            {
                checks[0] = false;
                checks[1] = false;
                checks[2] = false;
                checks[3] = false;
                check_count = 0;

                if ((j > 0) && (!hot[j - 1][k]) && (!dead[j - 1][k]))
                {
                    checks[0] = true;
                    check_count++;
                }
                if ((j < 31) && (!hot[j + 1][k]) && (!dead[j + 1][k]))
                {
                    checks[1] = true;
                    check_count++;
                }
                if ((k > 0) && (!hot[j][k - 1]) && (!dead[j][k - 1]))
                {
                    checks[2] = true;
                    check_count++;
                }
                if ((k < 31) && (!hot[j][k + 1]) && (!dead[j][k + 1]))
                {
                    checks[3] = true;
                    check_count++;
                }
                
                if (check_count == 0)
                {
                    continue;
                }

                for (int i = 0; i < current_fgs1_time_dim; i++)
                {
                    double sum = 0.0;
                    if (checks[0])
                    {
                        sum += temp_signal[i][j - 1][k];
                    }
                    if (checks[1])
                    {
                        sum += temp_signal[i][j + 1][k];
                    }
                    if (checks[2])
                    {
                        sum += temp_signal[i][j][k - 1];
                    }
                    if (checks[3])
                    {
                        sum += temp_signal[i][j][k + 1];
                    }

                    signal[i][j][k] = sum / check_count;
                }
            }
        }
    }
    free(temp_signal); 
}

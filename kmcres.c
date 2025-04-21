/*
    Kinetic Monte Carlo Algorithm based on Rejection Sampling in Glass System
    Made by Sanggeun Song, University of California, Berkeley.

    Prerequisites:
    - C compiler (gcc, clang, Visual studio, etc)
    - gsd.h and gsd.c (included, these files can be get from https://github.com/glotzerlab/gsd.git)

    (*) Linux
    - You can compile this file using this command:
    [user@localhost ~]$ gcc kmcres.c gsd.c -o kmcres.exe -lm (-march=native)

    (*) Mac
    - You can compile this file using this command:
    user@localhost ~ % clang kmcres.c gsd.c -o kmcres.exe

    (*) Windows
    - You can compile this file using Visual Studio:
    1) Launch Visual studio, make new empty C++ project.
    2) Add kmcres.c, gsd.c in source file
    3) Add gsd.h in header file
    4) Compile, and use it!
*/

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>
#include <signal.h>
#include <stdint.h>
#include "gsd.h"

static uint64_t state[4];
static inline uint64_t rotl(const uint64_t x, int k);
uint64_t xoshiro256p_next(void);
double uniform_random(void);
double bessel_J0(double x);
double bessel_I0(double x);
size_t Generate_lattice(double L, double ld, float **q);
int Sample_site(double *r, size_t size);
double Sample_angle(double kappa, double betaJ, double gamma_mu, double eta_mu);
void Handle_SIGINT(int sig);
void Handle_SIGTERM(int sig);

#define STRLEN 128
bool signal_caught = false;

int main(int argc, char *argv[])
{
    double *gamma, *eta, *r, *psi;
    float *persistence;
    float *q;
    uint32_t *typeid;

    if (argc != 6)
    {
        printf("Usage: %s (L) (kappa) (betaJ) (filename) (gsd_mode, [on|off])\n", argv[0]);
        printf("Example: %s 15 0.5 10/3 exc_1 on\n", argv[0]);
        exit(EXIT_SUCCESS);
    }

    // Input parameter (reading from command line input)
    int L = atoi(argv[1]);
    double kappa = atof(argv[2]);
    double betaJ = atof(argv[3]);
    char *filename_prefix = argv[4];
    char *gsd_mode_string = argv[5];

    // Input parameters
    double nu = 0.35;            // Poisson ratio
    double ld = 4.0 / 3.0;       // Lattice spacing
    double Rf = sqrt(3.0) / 2.0; // Radius of final excitation
    double udd = 0.116;          // u double dagger, JCP 155 044504 (2021), Table II
    double nu0 = 1.0;
    double A = 2.0 * M_PI * (sqrt(3.0) / 2) * (0.5 * ld / Rf) * (0.5 * ld / Rf);
    double ec = 2.0 * sqrt(2.0) * (udd * (1.0 + udd)) / (1.0 + 2.0 * udd); // Based on JCP 155 044504 (2021), Eq. (34)
    double ef = 2.0 * ec;
    double theta_factor = sqrt(2.0) * (Rf * Rf) * ef / (1.0 + nu);
    double disp_factor = (2.0 * Rf) * (2.0 * Rf) / (4.0 * sqrt(2.0) * (1.0 + nu)) * ef;

    // kMC simulation time and logging criteria
    double threshold = 0.01;
    double time_limit = 1e3;
    unsigned long long int threshold_limit = 10000;
    unsigned long long int log_interval = 1;

    char filename[STRLEN], filename_gsd[STRLEN], filename_restart[STRLEN], filename_finish[STRLEN];
    bool gsd_mode, reachth;
    FILE *fp, *fp_restart; // File pointer for log and restart file
    struct gsd_handle fp_gsd;
    size_t meshpoint = 0;
    size_t mu, mu_idx;
    double R, W, tk, tmarkov, P_t, q_dist, theta, Cb, MSD, Fsk, val, E;
    double *thetavals, *ux, *uy; // Fields
    float boxvec[6] = {(float)L * ld, (float)L * ld, 0.0, 0.0, 0.0};
    uint8_t particle_type[4] = {'A', '\0', 'B', '\0'};
    int flag;
    double t_before;

    unsigned long long int num_events = 0, num_threshold = 0, num_excitation = 0;

    // Magic number for restart file.
    unsigned long long int magic_number = 0xA5D64F13D48F2E78;

    // Set the filename
    strncpy(filename, filename_prefix, STRLEN - 5);
    strncpy(filename_gsd, filename_prefix, STRLEN - 5);
    strncpy(filename_restart, filename_prefix, STRLEN - 5);
    strncpy(filename_finish, filename_prefix, STRLEN - 5);
    strcat(filename, ".log");
    strcat(filename_gsd, ".gsd");
    strcat(filename_restart, ".bin");
    strcat(filename_finish, ".fin");

    // Set GSD mode
    if (strcmp(gsd_mode_string, "on") == 0)
        gsd_mode = true;
    else if (strcmp(gsd_mode_string, "off") == 0)
        gsd_mode = false;
    else
    {
        fprintf(stderr, "*** Fatal error: gsd_mode should be one of on or off.\n");
        exit(EXIT_FAILURE);
    }

    // If the simulation is already done, exit program.
    fp = fopen(filename_finish, "r");
    if (fp != NULL)
    {
        fprintf(stdout, "Simulation already finished! Filename: %s\n", filename);
        exit(EXIT_SUCCESS);
    }

    // Set up the signal handler for SIGINT and SIGTERM
    signal(SIGINT, Handle_SIGINT);
    signal(SIGTERM, Handle_SIGTERM);

    // Set the seed for random number generator
    state[0] = (uint64_t)time(NULL);
    state[1] = state[0] + 123456789;
    state[2] = state[1] + 987654321;
    state[3] = state[2] + 543219876;

    kappa = kappa * 4.0 * Rf * Rf;
    // Generate the lattice
    meshpoint = Generate_lattice(L * ld, ld, &q);
    // Set total rate
    W = nu0 * A * meshpoint * exp(-betaJ);
    // Initialize time
    tmarkov = 0.0;
    // Initialize persistence variable, P(t)
    P_t = 0.0;
    // Initialize threshold boolean variable
    reachth = false;

    typeid = (uint32_t *)malloc(meshpoint * sizeof(uint32_t));
    persistence = (float *)malloc(meshpoint * sizeof(float));
    gamma = (double *)malloc(meshpoint * sizeof(double));
    eta = (double *)malloc(meshpoint * sizeof(double));
    r = (double *)malloc(meshpoint * sizeof(double));
    psi = (double *)malloc(meshpoint * sizeof(double));
    thetavals = (double *)malloc(meshpoint * sizeof(double));
    ux = (double *)malloc(meshpoint * sizeof(double));
    uy = (double *)malloc(meshpoint * sizeof(double));

    for (mu_idx = 0; mu_idx < meshpoint; mu_idx++)
    {
        typeid[mu_idx] = 0;
        persistence[mu_idx] = 0;
        gamma[mu_idx] = 0.0;
        eta[mu_idx] = 0.0;
        thetavals[mu_idx] = 0.0;
        ux[mu_idx] = 0.0;
        uy[mu_idx] = 0.0;
        r[mu_idx] = 0.0;
    }

    fprintf(stdout, "--- L = %d ---\n", L);
    fprintf(stdout, "--- ld = %f ---\n", ld);
    fprintf(stdout, "--- betaJ = %f ---\n", betaJ);
    fprintf(stdout, "--- kappa = %f ---\n", kappa);
    fprintf(stdout, "--- Filename prefix = %s ---\n", filename_prefix);
    fprintf(stdout, "Start!\n");

    // Set restart mode
    fp_restart = fopen(filename_restart, "rb+");
    if (fp_restart != NULL)
    {
        // Read the information from restart file
        unsigned long long int magic_number_check;
        fread(&magic_number_check, sizeof(unsigned long long int), 1, fp_restart);
        if (magic_number_check != magic_number)
        {
            fprintf(stderr, "***Fatal error: corrupted restart file. Check your restart file %s\n", filename_restart);
            exit(EXIT_FAILURE);
        }
        fread(&num_events, sizeof(unsigned long long int), 1, fp_restart);
        fread(&num_threshold, sizeof(unsigned long long int), 1, fp_restart);
        fread(&tmarkov, sizeof(double), 1, fp_restart);
        fread(typeid, sizeof(uint32_t), meshpoint, fp_restart);
        fread(persistence, sizeof(float), meshpoint, fp_restart);
        fread(psi, sizeof(double), meshpoint, fp_restart);
        fread(gamma, sizeof(double), meshpoint, fp_restart);
        fread(eta, sizeof(double), meshpoint, fp_restart);
        fread(thetavals, sizeof(double), meshpoint, fp_restart);
        fread(ux, sizeof(double), meshpoint, fp_restart);
        fread(uy, sizeof(double), meshpoint, fp_restart);

        // Open the log file with append mode, and go to the last line
        fp = fopen(filename, "r+");
        if (fp == NULL)
        {
            fprintf(stderr, "***Fatal error: Failed to open log file %s\n", filename);
            exit(EXIT_FAILURE);
        }
        fseek(fp, 0, SEEK_END);

        t_before = tmarkov;

        // GSD writer
        if (gsd_mode)
        {
            flag = gsd_open(&fp_gsd, filename_gsd, GSD_OPEN_READWRITE);
            if (flag != 0)
            {
                fprintf(stderr, "***Fatal error: Failed to open gsd file %s\n", filename_gsd);
                exit(EXIT_FAILURE);
            }
            // This is just for check for validity of GSD file.
            // Note that it only checks number of particle and number of step!
            const char *chunk_name;
            const struct gsd_index_entry *gsd_data;
            uint32_t N;
            uint64_t steps, nframe;
            nframe = gsd_get_nframes(&fp_gsd);
            chunk_name = gsd_find_matching_chunk_name(&fp_gsd, "particles/N", NULL);
            gsd_data = gsd_find_chunk(&fp_gsd, 0, chunk_name);
            gsd_read_chunk(&fp_gsd, &N, gsd_data);
            if (N != meshpoint)
            {
                fprintf(stderr, "***Fatal error: Number of particles does not match!\n");
                exit(EXIT_FAILURE);
            }
            chunk_name = gsd_find_matching_chunk_name(&fp_gsd, "configuration/step", NULL);
            gsd_data = gsd_find_chunk(&fp_gsd, nframe - 1, chunk_name);
            flag = gsd_read_chunk(&fp_gsd, &steps, gsd_data);
            if (steps != num_events)
            {
                fprintf(stderr, "***Fatal error: Number of step does not match! flag: %d\n", flag);
                fprintf(stderr, "Number of steps in gsd: %llu, and in binary: %llu\n", steps, num_events);
                exit(EXIT_FAILURE);
            }
        }
        // Update P_t and num_excitation
        for (mu_idx = 0; mu_idx < meshpoint; mu_idx++)
        {
            P_t += persistence[mu_idx] / meshpoint;
            num_excitation += typeid[mu_idx];
        }

        printf("Note: Simulation will restart from tmarkov = %f\n", tmarkov);
    }
    else
    {
        fp = fopen(filename, "w");
        if (fp == NULL)
        {
            fprintf(stderr, "***Fatal error: Failed to open log file %s\n", filename);
            exit(EXIT_FAILURE);
        }

        fprintf(fp, "timestep tmarkov Cb Fsk MSD Pt N threshold");
        fprintf(fp, "\n0 0.0 1.0 1.0 0.0 0.0 0 False");
        t_before = 0.0;

        fp_restart = fopen(filename_restart, "wb");
        if (fp_restart == NULL)
        {
            fprintf(stderr, "***Fatal error: Failed to open restart file %s\n", filename_restart);
            exit(EXIT_FAILURE);
        }

        if (gsd_mode)
        {
            flag = gsd_create_and_open(&fp_gsd, filename_gsd, "KMC", "hoomd", gsd_make_version(1, 4), GSD_OPEN_READWRITE, 0);
            if (flag != 0)
            {
                fprintf(stderr, "***Fatal error: Failed to open gsd file %s\n", filename_gsd);
                exit(EXIT_FAILURE);
            }
            gsd_write_chunk(&fp_gsd, "particles/N", GSD_TYPE_UINT32, 1, 1, 0, &meshpoint);
            gsd_write_chunk(&fp_gsd, "particles/position", GSD_TYPE_FLOAT, meshpoint, 3, 0, q);
            gsd_write_chunk(&fp_gsd, "particles/diameter", GSD_TYPE_FLOAT, meshpoint, 1, 0, persistence);
            gsd_write_chunk(&fp_gsd, "particles/typeid", GSD_TYPE_UINT32, meshpoint, 1, 0, typeid);
            gsd_write_chunk(&fp_gsd, "configuration/box", GSD_TYPE_FLOAT, 6, 1, 0, boxvec);
            gsd_write_chunk(&fp_gsd, "configuration/step", GSD_TYPE_UINT64, 1, 1, 0, &num_events);
            gsd_write_chunk(&fp_gsd, "particles/types", GSD_TYPE_UINT8, 2, 2, 0, particle_type);
            // If you need to save psi or several fields, modify here. It may need to change some datatype.
            // gsd_write_chunk(fp_gsd, "particles/orientation", GSD_TYPE_DOUBLE, meshpoint, 3, 0, psi);
            // gsd_write_chunk(fp_gsd, "particles/charge", GSD_TYPE_DOUBLE, meshpoint, 1, 0, thetavals);
            // gsd_write_chunk(fp_gsd, "particles/mass", GSD_TYPE_DOUBLE, meshpoint, 1, 0, &r);
            // gsd_write_chunk(fp_gsd, "particles/velocity", GSD_TYPE_DOUBLE, meshpoint, 3, 0, &r);
            gsd_end_frame(&fp_gsd);
        }
    }

    // Do the KMC simulation!
    while (1)
    {
        num_events += 1;
        for (mu_idx = 0; mu_idx < meshpoint; mu_idx++)
        {
            if (typeid[mu_idx] == 1)
                r[mu_idx] = exp(kappa * betaJ * (gamma[mu_idx] * cos(2.0 * psi[mu_idx]) + eta[mu_idx] * sin(2.0 * psi[mu_idx])));
            else
                r[mu_idx] = A * bessel_I0(kappa * betaJ * sqrt(gamma[mu_idx] * gamma[mu_idx] + eta[mu_idx] * eta[mu_idx]));
        }
        R = 0.0;
        for (mu_idx = 0; mu_idx < meshpoint; mu_idx++)
            R += r[mu_idx];
        for (mu_idx = 0; mu_idx < meshpoint; mu_idx++)
            r[mu_idx] /= R;
        mu = Sample_site(r, meshpoint);
        if (typeid[mu] == 0)
        {
            typeid[mu] = 1; // Insert excitation
            E = 1.0;
            num_excitation += 1;
            if (persistence[mu] == 0)
            {
                persistence[mu] = 1; // Set persistence variable as 1
                P_t += 1.0 / meshpoint;
            }
            // Sample angle, psi[mu]
            psi[mu] = Sample_angle(kappa, betaJ, gamma[mu], eta[mu]);
        }
        else
        {
            typeid[mu] = 0; // Delete excitation
            E = -1.0;
            num_excitation -= 1;
        }
        tk = -log(1 - uniform_random()) / W;
        tmarkov += tk;
        // Update gamma and eta, and several fields for each excitation
        for (mu_idx = 0; mu_idx < meshpoint; mu_idx++)
        {
            if (mu_idx == mu)
            {
                theta = 0.0;
                q_dist = 0.25 * ld * ld;
                thetavals[mu_idx] -= theta_factor * E * sin(2.0 * theta - 2.0 * psi[mu]) / q_dist;
                ux[mu_idx] += E * (disp_factor * ((3.0 - nu) * 0.5 * cos(theta - 2.0 * psi[mu]) + 0.5 * (1.0 + nu) * cos(3.0 * theta - 2.0 * psi[mu]))) / sqrt(q_dist);
                uy[mu_idx] += E * (disp_factor * (-(3.0 - nu) * 0.5 * sin(theta - 2.0 * psi[mu]) + 0.5 * (1.0 + nu) * sin(3.0 * theta - 2.0 * psi[mu]))) / sqrt(q_dist);
                continue;
            }
            q_dist = (q[3 * mu] - q[3 * mu_idx]) * (q[3 * mu] - q[3 * mu_idx]) + (q[3 * mu + 1] - q[3 * mu_idx + 1]) * (q[3 * mu + 1] - q[3 * mu_idx + 1]);
            theta = atan2(q[3 * mu + 1] - q[3 * mu_idx + 1], q[3 * mu] - q[3 * mu_idx]);
            gamma[mu_idx] -= E * cos(4.0 * theta - 2.0 * psi[mu]) / q_dist;
            eta[mu_idx] -= E * sin(4.0 * theta - 2.0 * psi[mu]) / q_dist;
            thetavals[mu_idx] -= theta_factor * E * sin(2.0 * theta - 2.0 * psi[mu]) / q_dist;
            ux[mu_idx] += E * (disp_factor * ((3.0 - nu) * 0.5 * cos(theta - 2.0 * psi[mu]) + 0.5 * (1.0 + nu) * cos(3.0 * theta - 2.0 * psi[mu]))) / sqrt(q_dist);
            uy[mu_idx] += E * (disp_factor * (-(3.0 - nu) * 0.5 * sin(theta - 2.0 * psi[mu]) + 0.5 * (1.0 + nu) * sin(3.0 * theta - 2.0 * psi[mu]))) / sqrt(q_dist);
        }
        // Standard output control
        if (num_events % log_interval == 0)
        {
            fprintf(stdout, "\rtmarkov: %f of %f, num_threshold: %llu of %llu", tmarkov, time_limit, num_threshold, threshold_limit);
            fflush(stdout);
        }
        // Writing file
        if (num_events % log_interval == 0)
        {
            t_before = tmarkov;
            // Compute all relaxation measures
            Cb = 0.0;
            Fsk = 0.0;
            MSD = 0.0;
            for (mu_idx = 0; mu_idx < meshpoint; mu_idx++)
            {
                Cb += cos(6 * thetavals[mu_idx]) / meshpoint;
                val = ux[mu_idx] * ux[mu_idx] + uy[mu_idx] * uy[mu_idx];
                MSD += val / meshpoint;
                Fsk += bessel_J0(2.0 * M_PI * sqrt(val)) / meshpoint;
            }
            // Write log file
            if (reachth)
                fprintf(fp, "\n%llu %.15f %.15f %.15f %.15f %.15f %llu True", num_events, tmarkov, Cb, Fsk, MSD, P_t, num_excitation);
            else
                fprintf(fp, "\n%llu %.15f %.15f %.15f %.15f %.15f %llu False", num_events, tmarkov, Cb, Fsk, MSD, P_t, num_excitation);
            fflush(fp);

            // GSD writer
            if (gsd_mode)
            {
                gsd_write_chunk(&fp_gsd, "particles/diameter", GSD_TYPE_FLOAT, meshpoint, 1, 0, persistence);
                gsd_write_chunk(&fp_gsd, "particles/typeid", GSD_TYPE_UINT32, meshpoint, 1, 0, typeid);
                gsd_write_chunk(&fp_gsd, "configuration/step", GSD_TYPE_UINT64, 1, 1, 0, &num_events);
                gsd_end_frame(&fp_gsd);
                gsd_flush(&fp_gsd);
            }
            // Write restart file
            // We need to save magic_number, num_events, num_threshold, tmarkov, typeid, persistence, psi, gamma, eta, thetavals, ux, and uy for restart simulation!
            rewind(fp_restart);
            fwrite(&magic_number, sizeof(unsigned long long int), 1, fp_restart);
            fwrite(&num_events, sizeof(unsigned long long int), 1, fp_restart);
            fwrite(&num_threshold, sizeof(unsigned long long int), 1, fp_restart);
            fwrite(&tmarkov, sizeof(double), 1, fp_restart);
            fwrite(typeid, sizeof(uint32_t), meshpoint, fp_restart);
            fwrite(persistence, sizeof(float), meshpoint, fp_restart);
            fwrite(psi, sizeof(double), meshpoint, fp_restart);
            fwrite(gamma, sizeof(double), meshpoint, fp_restart);
            fwrite(eta, sizeof(double), meshpoint, fp_restart);
            fwrite(thetavals, sizeof(double), meshpoint, fp_restart);
            fwrite(ux, sizeof(double), meshpoint, fp_restart);
            fwrite(uy, sizeof(double), meshpoint, fp_restart);
            fflush(fp_restart);
        }
        // Stop the loop with criteria
        if (1.0 - P_t < threshold)
        {
            reachth = true;
            num_threshold += 1;
            if (num_threshold > threshold_limit)
                break;
        }
        if (tmarkov >= time_limit)
            break;
        // Stop the loop if program caught SIGINT or SIGTERM signal
        if (signal_caught)
            break;
    }

    // Close several files
    fclose(fp);
    fclose(fp_restart);
    if (gsd_mode)
        gsd_close(&fp_gsd);

    // Deallocate several arrays
    free(typeid);
    free(persistence);
    free(gamma);
    free(eta);
    free(r);
    free(psi);
    free(thetavals);
    free(ux);
    free(uy);
    free(q);

    if (signal_caught)
        printf("KMC simulation stopped before reaching criteria.\n");
    else
    {
        fp = fopen(filename_finish, "w");
        fclose(fp);
        printf("\nDone. KMC simulation will exit.\n");
    }
    return EXIT_SUCCESS;
}

// Random number generator (Xoshiro256++ algorithm)
static inline uint64_t rotl(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

uint64_t xoshiro256p_next(void)
{
    const uint64_t result = state[0] + state[3];
    const uint64_t t = state[1] << 17;

    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];

    state[2] ^= t;
    state[3] = rotl(state[3], 45);

    return result;
}

double uniform_random(void)
{
    // Convert the output of xoshiro256++ to a double in [0, 1)
    return (xoshiro256p_next() >> 11) * 0x1.0p-53;
}

// Bessel J0 function, based on Numerical Recipes in C
double bessel_J0(double x)
{
    double ax = fabs(x);
    double y, result;

    if (ax < 8.0)
    {
        y = x * x;
        result = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        result /= 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0))));
    }
    else
    {
        double z = 8.0 / ax;
        y = z * z;
        double xx = ax - 0.25 * M_PI;
        result = sqrt(2.0 / (M_PI * ax)) *
                 (cos(xx) * (1.0
                 + y * (-0.1098628627e-2
                 + y * (0.2734510407e-4
                 + y * (-0.2073370639e-5
                 + y * 0.2093887211e-6)))
                 - z * sin(xx) * (-0.1562499995e-1
                 + y * (0.1430488765e-3
                 + y * (-0.6911147651e-5
                 + y * (0.7621095161e-6
                 - y * 0.934935152e-7))))));
    }

    return result;
}

// Bessel I0 function, based on Numerical Recipes in C
double bessel_I0(double x)
{
    double ax = fabs(x);
    double y, result;

    if (ax < 3.75)
    {
        y = x / 3.75;
        y *= y;
        result = 1.0 + y * (3.5156229 +
                            y * (3.0899424 +
                                 y * (1.2067492 +
                                      y * (0.2659732 +
                                           y * (0.0360768 +
                                                y * 0.0045813)))));
    }
    else
    {
        y = 3.75 / ax;
        result = (exp(ax) / sqrt(ax)) * (0.39894228 +
                                         y * (0.01328592 +
                                              y * (0.00225319 +
                                                   y * (-0.00157565 +
                                                        y * (0.00916281 +
                                                             y * (-0.02057706 +
                                                                  y * (0.02635537 +
                                                                       y * (-0.01647633 +
                                                                            y * 0.00392377))))))));
    }
    return result;
}

size_t Generate_lattice(double L, double ld, float **q)
{
    double b = ld * sqrt(3.0) / 2.0;
    size_t Mx = (size_t)round(L / ld);
    size_t My = (size_t)round(L / b);

    double *x = (double *)malloc(Mx * My * sizeof(double));
    double *y = (double *)malloc(Mx * My * sizeof(double));

    size_t i, j;
    for (i = 0; i < Mx; i++)
    {
        for (j = 0; j < My; j++)
        {
            x[i * My + j] = i * ld + (j % 2) * ld / 2.0;
            y[i * My + j] = j * b;
        }
    }

    double xmean = 0.0, ymean = 0.0;
    for (i = 0; i < Mx * My; i++)
    {
        xmean += x[i] / (Mx * My);
        ymean += y[i] / (Mx * My);
    }

    for (i = 0; i < Mx * My; i++)
    {
        x[i] -= xmean;
        y[i] -= ymean;
    }
    *q = (float *)malloc(Mx * My * 3 * sizeof(float));
    for (i = 0; i < Mx * My; i++)
    {
        (*q)[3 * i] = x[i];
        (*q)[3 * i + 1] = y[i];
        (*q)[3 * i + 2] = 0.0;
    }

    free(x);
    free(y);
    return (size_t)(Mx * My);
}

int Sample_site(double *r, size_t size)
{
    double rn = uniform_random();
    double cumulative = 0.0;
    size_t i;
    for (i = 0; i < size; i++)
    {
        cumulative += r[i];
        if (rn < cumulative)
            return i;
    }
    return size - 1;
}

double Sample_angle(double kappa, double betaJ, double gamma_mu, double eta_mu)
{
    double alpha = betaJ * kappa * sqrt(gamma_mu * gamma_mu + eta_mu * eta_mu);
    double rho = 2.0 * alpha / (1.0 + sqrt(1.0 + 4.0 * alpha * alpha));
    double u, v, y, z, phi, psi;
    while (1)
    {
        u = uniform_random();
        v = uniform_random();
        y = cos(2.0 * M_PI * v);
        z = 1.0 / (1.0 - rho * y);
        if (u <= z * exp(1.0 - z))
            break;
    }
    if (v < 0.5)
        phi = acos(z * (rho - y));
    else
        phi = -acos(z * (rho - y));
    psi = (phi + atan2(eta_mu, gamma_mu)) / 2.0;
    if (psi < 0)
        psi += M_PI;
    return psi;
}

void Handle_SIGINT(int sig)
{
    fprintf(stderr, "\nCaught SIGINT (signal %d)\n", sig);
    fprintf(stderr, "Please wait until saving all files...\n");
    signal_caught = true;
}

void Handle_SIGTERM(int sig)
{
    fprintf(stderr, "\nCaught SIGTERM (signal %d)\n", sig);
    fprintf(stderr, "Please wait until saving all files...\n");
    signal_caught = true;
}
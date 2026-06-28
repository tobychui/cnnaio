/*
 * classify.c — ncnn inference driver compiled to wasm32-wasi.
 *
 * Everything runs *inside* the wasm sandbox: image decoding (stb_image),
 * preprocessing, and ncnn inference. The host (wazero) mounts the model files
 * and the input image as read-only filesystems and passes everything else as argv.
 *
 * Diagnostics always go to stderr; stdout carries only the machine-readable
 * result of the chosen subcommand (argv[1]):
 *
 *   classify  — image classification, prints PRED lines (see cmd_classify).
 *   infer     — generic: run a net and dump named output tensors as binary
 *               (see cmd_infer). Used by the object detectors, which decode the
 *               raw tensors (anchors + NMS) on the Go side.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ncnn/c_api.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static char* read_text_line(FILE* fp)
{
    static char buf[1024];
    if (!fgets(buf, sizeof(buf), fp)) return NULL;
    size_t n = strlen(buf);
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = 0;
    return buf;
}

/* Decode an image file to RGB, optionally crop a region of interest (roiw/roih>0),
 * resize to target, optionally swap to BGR, and apply mean/normalize. Returns a
 * new ncnn_mat (caller destroys) or NULL. The image's original pixel dimensions
 * are returned via *orig_w / *orig_h. */
static ncnn_mat_t load_and_preprocess(const char* image_path, int target_w, int target_h,
                                      int to_bgr, const float mean[3], const float norm[3],
                                      int roix, int roiy, int roiw, int roih,
                                      int* orig_w, int* orig_h)
{
    int w = 0, h = 0, ch = 0;
    unsigned char* pixels = stbi_load(image_path, &w, &h, &ch, 3); /* force RGB */
    if (!pixels)
    {
        fprintf(stderr, "failed to decode image: %s (%s)\n", image_path, stbi_failure_reason());
        return NULL;
    }
    *orig_w = w;
    *orig_h = h;

    int pixel_type = to_bgr ? NCNN_MAT_PIXEL_X2Y(NCNN_MAT_PIXEL_RGB, NCNN_MAT_PIXEL_BGR)
                            : NCNN_MAT_PIXEL_RGB;

    ncnn_mat_t in;
    if (roiw > 0 && roih > 0)
    {
        /* clamp the ROI inside the image */
        if (roix < 0) { roiw += roix; roix = 0; }
        if (roiy < 0) { roih += roiy; roiy = 0; }
        if (roix + roiw > w) roiw = w - roix;
        if (roiy + roih > h) roih = h - roiy;
        fprintf(stderr, "[wasm] decoded %dx%d -> roi %d,%d %dx%d -> resize %dx%d\n",
                w, h, roix, roiy, roiw, roih, target_w, target_h);
        in = ncnn_mat_from_pixels_roi_resize(pixels, pixel_type, w, h, w * 3,
                                             roix, roiy, roiw, roih, target_w, target_h, NULL);
    }
    else
    {
        fprintf(stderr, "[wasm] decoded image %dx%d (%d ch) -> resize %dx%d\n", w, h, ch, target_w, target_h);
        in = ncnn_mat_from_pixels_resize(pixels, pixel_type, w, h, w * 3,
                                         target_w, target_h, NULL);
    }
    stbi_image_free(pixels);
    if (!in) return NULL;
    ncnn_mat_substract_mean_normalize(in, mean, norm);
    return in;
}

/* ----------------------------------------------------------------------------
 * classify — binary param, single output, softmax + top-k. Prints to stdout:
 *     PRED \t rank \t index \t score(0..1) \t label
 * (argv here is shifted so argv[1] is the first real arg, as before.)
 * -------------------------------------------------------------------------- */
static int cmd_classify(int argc, char** argv)
{
    if (argc < 14)
    {
        fprintf(stderr, "usage: classify param model synset image tw th tobgr m0 m1 m2 n0 n1 n2 [topk]\n");
        return 2;
    }

    const char* param_path  = argv[1];
    const char* model_path  = argv[2];
    const char* synset_path = argv[3];
    const char* image_path  = argv[4];
    int target_w = atoi(argv[5]);
    int target_h = atoi(argv[6]);
    int to_bgr   = atoi(argv[7]);
    float mean_vals[3] = { (float)atof(argv[8]),  (float)atof(argv[9]),  (float)atof(argv[10]) };
    float norm_vals[3] = { (float)atof(argv[11]), (float)atof(argv[12]), (float)atof(argv[13]) };
    int topk = (argc > 14) ? atoi(argv[14]) : 5;

    ncnn_net_t net = ncnn_net_create();
    ncnn_option_t opt = ncnn_option_create();
    ncnn_option_set_num_threads(opt, 1);
    ncnn_net_set_option(net, opt);

    if (ncnn_net_load_param_bin(net, param_path) != 0)
    {
        fprintf(stderr, "failed to load param: %s\n", param_path);
        return 1;
    }
    if (ncnn_net_load_model(net, model_path) != 0)
    {
        fprintf(stderr, "failed to load model: %s\n", model_path);
        return 1;
    }

    /* binary param has no blob names -> resolve input/output by index */
    int in_index  = ncnn_net_get_input_index(net, 0);
    int out_index = ncnn_net_get_output_index(net, 0);
    fprintf(stderr, "[wasm] input_index=%d output_index=%d\n", in_index, out_index);

    int orig_w, orig_h;
    ncnn_mat_t in = load_and_preprocess(image_path, target_w, target_h, to_bgr, mean_vals, norm_vals,
                                        0, 0, 0, 0, &orig_w, &orig_h);
    if (!in) return 1;

    ncnn_extractor_t ex = ncnn_extractor_create(net);
    ncnn_extractor_set_option(ex, opt);
    ncnn_extractor_input_index(ex, in_index, in);

    ncnn_mat_t out = NULL;
    if (ncnn_extractor_extract_index(ex, out_index, &out) != 0 || !out)
    {
        fprintf(stderr, "extract failed\n");
        return 1;
    }

    int ow = ncnn_mat_get_w(out);
    int oh = ncnn_mat_get_h(out);
    int oc = ncnn_mat_get_c(out);
    int n = ow * (oh > 0 ? oh : 1) * (oc > 0 ? oc : 1);
    const float* scores = (const float*)ncnn_mat_get_data(out);
    fprintf(stderr, "[wasm] output dims=%d w=%d h=%d c=%d -> %d scores\n",
            ncnn_mat_get_dims(out), ow, oh, oc, n);

    double rawsum = 0.0;
    int all_nonneg = 1;
    for (int i = 0; i < n; i++) { rawsum += scores[i]; if (scores[i] < 0.0f) all_nonneg = 0; }
    int already_prob = all_nonneg && (rawsum > 0.95 && rawsum < 1.05);

    float* prob = (float*)malloc(sizeof(float) * n);
    if (already_prob)
    {
        fprintf(stderr, "[wasm] output is already a probability distribution (sum=%.3f)\n", rawsum);
        for (int i = 0; i < n; i++) prob[i] = scores[i];
    }
    else
    {
        fprintf(stderr, "[wasm] output looks like logits (sum=%.3f) -> applying softmax\n", rawsum);
        float maxv = scores[0];
        for (int i = 1; i < n; i++) if (scores[i] > maxv) maxv = scores[i];
        double sum = 0.0;
        for (int i = 0; i < n; i++) { prob[i] = (float)exp((double)(scores[i] - maxv)); sum += prob[i]; }
        for (int i = 0; i < n; i++) prob[i] = (float)(prob[i] / sum);
    }

    if (topk > n) topk = n;
    int* used = (int*)calloc(n, sizeof(int));

    char** labels = (char**)calloc(n, sizeof(char*));
    FILE* sf = fopen(synset_path, "r");
    if (sf)
    {
        for (int i = 0; i < n; i++)
        {
            char* line = read_text_line(sf);
            if (!line) break;
            labels[i] = strdup(line);
        }
        fclose(sf);
    }

    fprintf(stderr, "[wasm] === Top-%d predictions ===\n", topk);
    for (int k = 0; k < topk; k++)
    {
        int best = -1;
        float bestv = -1.0f;
        for (int i = 0; i < n; i++)
            if (!used[i] && prob[i] > bestv) { bestv = prob[i]; best = i; }
        if (best < 0) break;
        used[best] = 1;
        const char* label = labels[best] ? labels[best] : "(no-label)";
        printf("PRED\t%d\t%d\t%.6f\t%s\n", k + 1, best, (double)bestv, label);
        fprintf(stderr, "[wasm] %2d. [%4d] %6.2f%%  %s\n", k + 1, best, bestv * 100.0f, label);
    }

    free(prob);
    free(used);
    ncnn_mat_destroy(in);
    ncnn_mat_destroy(out);
    ncnn_extractor_destroy(ex);
    ncnn_option_destroy(opt);
    ncnn_net_destroy(net);
    return 0;
}

/* ----------------------------------------------------------------------------
 * infer — generic: run a net and dump named output tensors as raw float32.
 *
 * argv (shifted, argv[1] = first real arg):
 *   1  param path
 *   2  model path
 *   3  param_is_text  (1 = text .param via load_param, 0 = binary .param.bin)
 *   4  image path
 *   5  target_w
 *   6  target_h
 *   7  to_bgr
 *   8 9 10   mean
 *   11 12 13 norm
 *   14 input_blob_name
 *   15 n_outputs
 *   16.. output blob names
 *
 * stdout protocol (binary):
 *   line:  "INFER <n_outputs> <orig_w> <orig_h>\n"
 *   then n lines: "T <name> <w> <h> <c>\n"
 *   then the raw payload: for each tensor in order, c*(h*w) float32 little-endian
 *        (channels concatenated, ncnn cstep padding removed).
 * -------------------------------------------------------------------------- */
static int cmd_infer(int argc, char** argv)
{
    if (argc < 20)
    {
        fprintf(stderr, "usage: infer param model paramIsText image tw th tobgr m0 m1 m2 n0 n1 n2 roix roiy roiw roih inName nOut outName...\n");
        return 2;
    }

    const char* param_path = argv[1];
    const char* model_path = argv[2];
    int param_is_text      = atoi(argv[3]);
    const char* image_path = argv[4];
    int target_w = atoi(argv[5]);
    int target_h = atoi(argv[6]);
    int to_bgr   = atoi(argv[7]);
    float mean_vals[3] = { (float)atof(argv[8]),  (float)atof(argv[9]),  (float)atof(argv[10]) };
    float norm_vals[3] = { (float)atof(argv[11]), (float)atof(argv[12]), (float)atof(argv[13]) };
    int roix = atoi(argv[14]);
    int roiy = atoi(argv[15]);
    int roiw = atoi(argv[16]);
    int roih = atoi(argv[17]);
    const char* in_name = argv[18];
    int n_out = atoi(argv[19]);
    if (n_out < 1 || argc < 20 + n_out)
    {
        fprintf(stderr, "infer: bad n_outputs (%d) / not enough names\n", n_out);
        return 2;
    }
    char** out_names = argv + 20;

    ncnn_net_t net = ncnn_net_create();
    ncnn_option_t opt = ncnn_option_create();
    ncnn_option_set_num_threads(opt, 1);
    ncnn_net_set_option(net, opt);

    int load_err = param_is_text ? ncnn_net_load_param(net, param_path)
                                 : ncnn_net_load_param_bin(net, param_path);
    if (load_err != 0)
    {
        fprintf(stderr, "failed to load param: %s\n", param_path);
        return 1;
    }
    if (ncnn_net_load_model(net, model_path) != 0)
    {
        fprintf(stderr, "failed to load model: %s\n", model_path);
        return 1;
    }

    int orig_w, orig_h;
    ncnn_mat_t in = load_and_preprocess(image_path, target_w, target_h, to_bgr, mean_vals, norm_vals,
                                        roix, roiy, roiw, roih, &orig_w, &orig_h);
    if (!in) return 1;

    ncnn_extractor_t ex = ncnn_extractor_create(net);
    ncnn_extractor_set_option(ex, opt);
    ncnn_extractor_input(ex, in_name, in);

    ncnn_mat_t* outs = (ncnn_mat_t*)calloc(n_out, sizeof(ncnn_mat_t));
    for (int i = 0; i < n_out; i++)
    {
        ncnn_mat_t m = NULL;
        if (ncnn_extractor_extract(ex, out_names[i], &m) != 0 || !m)
        {
            fprintf(stderr, "infer: extract '%s' failed\n", out_names[i]);
            return 1;
        }
        outs[i] = m;
    }

    /* header */
    printf("INFER %d %d %d\n", n_out, orig_w, orig_h);
    for (int i = 0; i < n_out; i++)
    {
        int w = ncnn_mat_get_w(outs[i]);
        int h = ncnn_mat_get_h(outs[i]);
        int c = ncnn_mat_get_c(outs[i]);
        printf("T %s %d %d %d\n", out_names[i], w, h, c);
        fprintf(stderr, "[wasm] tensor %s dims=%d w=%d h=%d c=%d\n",
                out_names[i], ncnn_mat_get_dims(outs[i]), w, h, c);
    }

    /* payload: raw float32, channel by channel (drops cstep padding) */
    for (int i = 0; i < n_out; i++)
    {
        int w = ncnn_mat_get_w(outs[i]);
        int h = ncnn_mat_get_h(outs[i]);
        int c = ncnn_mat_get_c(outs[i]);
        int plane = w * (h > 0 ? h : 1);
        for (int ci = 0; ci < (c > 0 ? c : 1); ci++)
        {
            const float* p = (const float*)ncnn_mat_get_channel_data(outs[i], ci);
            fwrite(p, sizeof(float), (size_t)plane, stdout);
        }
    }
    fflush(stdout);

    for (int i = 0; i < n_out; i++) ncnn_mat_destroy(outs[i]);
    free(outs);
    ncnn_mat_destroy(in);
    ncnn_extractor_destroy(ex);
    ncnn_option_destroy(opt);
    ncnn_net_destroy(net);
    return 0;
}

int main(int argc, char** argv)
{
    fprintf(stderr, "[wasm] ncnn version: %s\n", ncnn_version());
    if (argc < 2)
    {
        fprintf(stderr, "usage: %s <classify|infer> ...\n", argv[0]);
        return 2;
    }
    const char* cmd = argv[1];
    if (strcmp(cmd, "classify") == 0) return cmd_classify(argc - 1, argv + 1);
    if (strcmp(cmd, "infer") == 0)    return cmd_infer(argc - 1, argv + 1);

    fprintf(stderr, "unknown command: %s\n", cmd);
    return 2;
}

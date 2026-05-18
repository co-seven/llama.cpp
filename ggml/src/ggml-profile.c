#include "ggml-profile.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef GGML_BUILD_PROFILE
#    if defined(_WIN32)
int g_current_token_idx = -1;

void ggml_set_current_token_idx(int idx) {
    g_current_token_idx = idx;
}

void ggml_profile_init_trace_file(void) {
}

void ggml_trace_log_begin(const char * name, const char * cat, const char * args) {
    GGML_UNUSED(name);
    GGML_UNUSED(cat);
    GGML_UNUSED(args);
}

void ggml_trace_log_end(const char * name, const char * cat, const char * args) {
    GGML_UNUSED(name);
    GGML_UNUSED(cat);
    GGML_UNUSED(args);
}

void ggml_profile_log_op_begin(struct ggml_tensor * t) {
    GGML_UNUSED(t);
}

void ggml_profile_log_op_end(struct ggml_tensor * t) {
    GGML_UNUSED(t);
}

void ggml_profile_flush_tls(void) {
}

void ggml_profile_flush_trace(void) {
}
#    else
#        include <pthread.h>
#        include <time.h>
#        if defined(__linux__)
#            include <sys/syscall.h>
#            include <unistd.h>
#        endif

static int            g_trace_enabled = 0;
static pthread_once_t g_trace_once    = PTHREAD_ONCE_INIT;

static void ggml_trace_parse_env(void) {
    const char * env = getenv("GGML_TRACE");
    if (env && (env[0] == '1' || env[0] == 'y' || env[0] == 'Y')) {
        g_trace_enabled = 1;
    } else {
        g_trace_enabled = 0;
    }
}

#    define TRACE_ENABLED() (pthread_once(&g_trace_once, ggml_trace_parse_env), g_trace_enabled)

static inline int64_t ggml_trace_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
}

static inline unsigned long ggml_trace_tid(void) {
#        if defined(__linux__)
    return (unsigned long) syscall(SYS_gettid);
#        else
    return (unsigned long) (uintptr_t) pthread_self();
#        endif
}

static pthread_mutex_t g_trace_mutex = PTHREAD_MUTEX_INITIALIZER;
static FILE *          g_trace_file  = NULL;

#    define TRACE_BUFFER_SIZE (64 * 1024)

__thread char   g_trace_tls_buffer[TRACE_BUFFER_SIZE];
__thread size_t g_trace_tls_offset = 0;

int g_current_token_idx = -1;

void ggml_set_current_token_idx(int idx) {
    g_current_token_idx = idx;
}

static void ggml_profile_init_trace_file_locked(void) {
    if (!g_trace_file) {
        g_trace_file = fopen("ggml_trace.json", "w");
        if (g_trace_file) {
            fputs("[\n", g_trace_file);
        }
    }
}

void ggml_profile_init_trace_file(void) {
    if (!TRACE_ENABLED()) {
        return;
    }

    pthread_mutex_lock(&g_trace_mutex);
    ggml_profile_init_trace_file_locked();
    pthread_mutex_unlock(&g_trace_mutex);
}

static void ggml_json_escape(const char * src, char * dst, size_t dst_size) {
    size_t i = 0;
    for (; *src && i + 2 < dst_size; src++) {
        if (*src == '"' || *src == '\\') {
            dst[i++] = '\\';
        }
        dst[i++] = *src;
    }
    dst[i] = '\0';
}

static inline const char * ggml_tensor_name_safe(struct ggml_tensor * t) {
    return (t && t->name[0] != '\0') ? t->name : "N/A";
}

static void ggml_format_tensor_info(struct ggml_tensor * t, const char * prefix, char * buf, size_t buf_size) {
    if (!t) {
        snprintf(buf, buf_size, "\"%s_exists\":false", prefix);
        return;
    }

    char escaped_name[64];
    ggml_json_escape(ggml_tensor_name_safe(t), escaped_name, sizeof(escaped_name));

    snprintf(buf, buf_size,
             "\"%s_shape\":\"[%ld,%ld,%ld,%ld]\","
             "\"%s_strides\":\"[%ld,%ld,%ld,%ld]\","
             "\"%s_type\":\"%s\","
             "\"%s_name\":\"%s\"",
             prefix, t->ne[0], t->ne[1], t->ne[2], t->ne[3],
             prefix, t->nb[0], t->nb[1], t->nb[2], t->nb[3],
             prefix, ggml_type_name(t->type), prefix, escaped_name);
}

static void ggml_format_op_args(struct ggml_tensor * t, const char * exec_path, char * buf, size_t buf_size) {
    char out_info[256];
    char src0_info[256];
    char src1_info[256];

    ggml_format_tensor_info(t, "out", out_info, sizeof(out_info));
    ggml_format_tensor_info(t->src[0], "src0", src0_info, sizeof(src0_info));
    ggml_format_tensor_info(t->src[1], "src1", src1_info, sizeof(src1_info));

    char op_name[64];
    ggml_json_escape(ggml_op_name(t->op), op_name, sizeof(op_name));

    if (exec_path) {
        char escaped_path[256];
        ggml_json_escape(exec_path, escaped_path, sizeof(escaped_path));

        snprintf(buf, buf_size, "\"op_name\":\"%s\",\"exec_path\":\"%s\",%s,%s,%s", op_name, escaped_path, out_info,
                 src0_info, src1_info);
    } else {
        snprintf(buf, buf_size, "\"op_name\":\"%s\",%s,%s,%s", op_name, out_info, src0_info, src1_info);
    }
}

static inline void ggml_trace_tls_flush_locked(void) {
    if (g_trace_tls_offset == 0 || !g_trace_file) {
        return;
    }

    fwrite(g_trace_tls_buffer, 1, g_trace_tls_offset, g_trace_file);
    g_trace_tls_offset = 0;
}

static inline void ggml_trace_write(const char * json) {
    if (!TRACE_ENABLED()) {
        return;
    }

    size_t len = strlen(json);

    if (g_trace_tls_offset + len >= TRACE_BUFFER_SIZE) {
        pthread_mutex_lock(&g_trace_mutex);
        ggml_profile_init_trace_file_locked();
        ggml_trace_tls_flush_locked();
        pthread_mutex_unlock(&g_trace_mutex);
    }

    memcpy(g_trace_tls_buffer + g_trace_tls_offset, json, len);
    g_trace_tls_offset += len;
}

void ggml_trace_log_begin(const char * name, const char * cat, const char * args) {
    if (!TRACE_ENABLED()) {
        return;
    }

    char buf[1024];

    snprintf(buf, sizeof(buf),
             "{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"B\","
             "\"ts\":%ld,\"pid\":1,\"tid\":%lu,\"args\":{%s}},\n",
             name, cat, ggml_trace_now_us(), ggml_trace_tid(), args ? args : "");

    ggml_trace_write(buf);
}

void ggml_trace_log_end(const char * name, const char * cat, const char * args) {
    if (!TRACE_ENABLED()) {
        return;
    }

    char buf[1024];

    snprintf(buf, sizeof(buf),
             "{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"E\","
             "\"ts\":%ld,\"pid\":1,\"tid\":%lu,\"args\":{%s}},\n",
             name, cat, ggml_trace_now_us(), ggml_trace_tid(), args ? args : "");

    ggml_trace_write(buf);
}

void ggml_profile_log_op_begin(struct ggml_tensor * t) {
    if (!TRACE_ENABLED()) {
        return;
    }

    char args[1024];
    ggml_format_op_args(t, NULL, args, sizeof(args));

    char name[128], tname[64];
    ggml_json_escape(ggml_tensor_name_safe(t), tname, sizeof(tname));

    snprintf(name, sizeof(name), "%s (%s)", ggml_op_name(t->op), tname);

    ggml_trace_log_begin(name, "Operator", args);
}

void ggml_profile_log_op_end(struct ggml_tensor * t) {
    if (!TRACE_ENABLED()) {
        return;
    }

    char name[128], tname[64];
    ggml_json_escape(ggml_tensor_name_safe(t), tname, sizeof(tname));

    snprintf(name, sizeof(name), "%s (%s)", ggml_op_name(t->op), tname);

    ggml_trace_log_end(name, "Operator", NULL);
}

void ggml_profile_flush_tls(void) {
    if (!TRACE_ENABLED()) {
        return;
    }

    pthread_mutex_lock(&g_trace_mutex);
    ggml_profile_init_trace_file_locked();
    ggml_trace_tls_flush_locked();
    pthread_mutex_unlock(&g_trace_mutex);
}

void ggml_profile_flush_trace(void) {
    if (!TRACE_ENABLED()) {
        return;
    }

    pthread_mutex_lock(&g_trace_mutex);

    if (g_trace_file) {
        ggml_trace_tls_flush_locked();

        long pos = ftell(g_trace_file);
        if (pos >= 2) {
            fseek(g_trace_file, pos - 2, SEEK_SET);  // remove ",\n"
        }

        fputs("\n]\n", g_trace_file);
        fclose(g_trace_file);
        g_trace_file = NULL;
    }

    pthread_mutex_unlock(&g_trace_mutex);
}
#    endif
#endif

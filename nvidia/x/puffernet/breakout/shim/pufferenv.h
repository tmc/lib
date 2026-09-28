// NVRTC stand-in for PufferLib's src/pufferenv.h. It defines the Agent
// struct, which is part of Env and must match the real one, makes the raylib
// calls in breakout.cu's host code no-ops, and implements dict_get, so that
// the kernel breakout_config can run gpu_breakout_host_fill_config.
// See cuda_runtime.h.
#pragma once

// exit sets exit_code and returns. It is here, not in stdlib.h, so that
// cudabind, which parses with the system stdlib.h, sees exit_code.
__device__ int exit_code;
#define exit(code) ((void)(exit_code = (code)))

// DictItem is a key and its value, as in PufferLib's src/ini.h.
typedef struct DictItem {
    char key[32];
    double value;
} DictItem;

typedef struct Dict {
    DictItem* items;
    int size;
} Dict;

// dict_get returns the value of key, or exits if dict has none.
static inline double dict_get(Dict* dict, const char* key) {
    for (int i = 0; i < dict->size; i++) {
        const char* k = dict->items[i].key;
        int j = 0;
        while (k[j] == key[j] && key[j] != 0) {
            j++;
        }
        if (k[j] == key[j]) {
            return dict->items[i].value;
        }
    }
    exit(1);
    return 0;
}

#define dict_set(dict, key, value) ((void)0)

typedef struct Env Env;
typedef struct Log Log;

typedef struct Agent {
    obs_t* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* action_mask;
    int policy;
} Agent;

#define PUF_CPU 0
#define PUF_GPU 1

#define puf_web_vsync() ((void)0)

typedef struct Color { unsigned char r, g, b, a; } Color;
typedef struct Rectangle { float x, y, width, height; } Rectangle;
typedef struct Vector2 { float x, y; } Vector2;
typedef struct Texture2D { unsigned int id; int width, height, mipmaps, format; } Texture2D;
#define RED {230, 41, 55, 255}
#define ORANGE {255, 161, 0, 255}
#define YELLOW {253, 249, 0, 255}
#define GREEN {0, 228, 48, 255}
#define SKYBLUE {102, 191, 255, 255}
#define BLUE {0, 121, 241, 255}
#define WHITE ((void)0)
#define KEY_TAB 0
#define KEY_ESCAPE 0
#define InitWindow(width, height, title) ((void)0)
#define CloseWindow() ((void)0)
#define SetTargetFPS(fps) ((void)0)
#define ToggleFullscreen() ((void)0)
#define IsKeyDown(key) false
#define IsKeyPressed(key) false
#define LoadTexture(path) Texture2D{}
#define UnloadTexture(...) ((void)0)
#define BeginDrawing() ((void)0)
#define EndDrawing() ((void)0)
#define ClearBackground(...) ((void)0)
#define DrawRectangle(...) ((void)0)
#define DrawTexturePro(...) ((void)0)
#define DrawText(...) ((void)0)
#define TextFormat(...) ""

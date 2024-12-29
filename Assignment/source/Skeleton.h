#pragma once
#include <glad.h>
#include <glfw3.h>

#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>

#include "filesystem_local.h"
#include "shader_m.h"
#include "camera.h"
#include "animator.h"
#include "model_animation.h"

extern void processModelAnimation();
extern Shader ourShader("anim_model.vs", "anim_model.fs");
extern Model ourModel(FileSystem::getPath("resources/objects/vampire/dancing_vampire.dae"));
extern Animation danceAnimation(FileSystem::getPath("resources/objects/vampire/dancing_vampire.dae"), &ourModel);
extern Animator animator(&danceAnimation);

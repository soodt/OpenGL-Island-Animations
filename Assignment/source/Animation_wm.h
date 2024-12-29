#pragma once
#include "glad.h"
#include "glfw3.h"
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <assimp/Importer.hpp> // Include Assimp header
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// Shaders
extern const char* vertexShaderSourceAnim = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    out vec3 FragPos;
    out vec3 Normal;
    void main()
    {
       FragPos = vec3(model * vec4(aPos, 1.0));
       Normal = mat3(transpose(inverse(model))) * aNormal;
       gl_Position = projection * view * model * vec4(aPos, 1.0);
    }\0";
)";

extern const char* fragmentShaderSourceAnim = R"(
    #version 330 core
    in vec3 FragPos;
    in vec3 Normal;
    out vec4 FragColor;
    void main()
    {
       // Implement basic lighting here (optional)
       vec3 lightPos = vec3(1.0f, 1.0f, 1.0f);
       vec3 lightDir = normalize(lightPos - FragPos);
       float diff = max(dot(Normal, lightDir), 0.0f);
       FragColor = vec4(diff, diff, diff, 1.0f);
    }\0";
)";

// Process the nodes
extern std::vector<unsigned int> indices;
extern std::vector<glm::vec3> vertices;
extern std::vector<glm::vec3> normals;

extern void LoadModel(const char* path, unsigned int& VAO, unsigned int& VBO, unsigned int& EBO);
extern void processNode(aiNode* node, const aiScene* scene, std::vector<unsigned int>& indices, std::vector<glm::vec3>& vertices, std::vector<glm::vec3>& normals);
extern void processMesh(aiMesh* mesh, const aiScene* scene, std::vector<unsigned int>& indices, std::vector<glm::vec3>& vertices, std::vector<glm::vec3>& normals);

// Function to load a GLTF model using Assimp
void LoadModel(const char* path, unsigned int& VAO, unsigned int& VBO, unsigned int& EBO) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cout << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
        return;
    }

    processNode(scene->mRootNode, scene, indices, vertices, normals);

    // Generate and bind VAO, VBO, and EBO
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

    // Vertex positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    // Vertex normals
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

// Recursive function to process Assimp's node hierarchy
void processNode(aiNode* node, const aiScene* scene, std::vector<unsigned int>& indices, std::vector<glm::vec3>& vertices, std::vector<glm::vec3>& normals) {
    // Process all the node's meshes (if any)
    for (unsigned int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        processMesh(mesh, scene, indices, vertices, normals);
    }
    // Then do the same for each of its children
    for (unsigned int i = 0; i < node->mNumChildren; i++) {
        processNode(node->mChildren[i], scene, indices, vertices, normals);
    }
}

// Function to process Assimp's mesh data
void processMesh(aiMesh* mesh, const aiScene* scene, std::vector<unsigned int>& indices, std::vector<glm::vec3>& vertices, std::vector<glm::vec3>& normals) {
    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
        vertices.push_back(glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z));
        normals.push_back(glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z));
    }

    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
        aiFace face = mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; j++) {
            indices.push_back(face.mIndices[j]);
        }
    }
}
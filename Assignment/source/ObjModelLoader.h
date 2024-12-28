#pragma once

#include "glm.hpp"
#include "gtc/matrix_transform.hpp"
#include "gtc/type_ptr.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;
};

class Model {
public:
    std::vector<Vertex> vertices;
    unsigned int VAO, VBO;
    unsigned int textureID;
    std::string modelPath;

    Model(const std::string& path, const std::string& texturePath) : modelPath(path) {
        loadModel(path, texturePath);
    }

    ~Model() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteTextures(1, &textureID);
    }

    void draw(unsigned int shaderProgram) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, vertices.size());
    }

private:
    bool loadModel(const std::string& path, const std::string& texturePath) {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn;
        std::string err;
        std::string mtl_basedir = std::filesystem::path(path).parent_path().string() + "/";

        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials,  &err, path.c_str(), mtl_basedir.c_str());

        if (!warn.empty()) {
            std::cerr << warn << std::endl;
        }

        if (!err.empty()) {
            std::cerr << err << std::endl;
        }

        if (!ret) {
            std::cerr << "Failed to load/parse .obj file" << std::endl;
            return false;
        }

        if (attrib.vertices.empty()) {
            std::cerr << "Error: No vertices found in OBJ file." << std::endl;
            return false;
        }

        
        for (size_t s = 0; s < shapes.size(); s++) {
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
                int fv = shapes[s].mesh.num_face_vertices[f];
                if (fv != 3) {
                    std::cerr << "Error: Only triangle faces are supported." << std::endl;
                    return false;
                }
                std::vector<glm::vec3> face_vertices(3); // Store vertices of the face
                for (size_t v = 0; v < fv; v++) {
                    tinyobj::index_t idx = shapes[s].mesh.indices[3 * f + v];
                    face_vertices[v] = {
                        attrib.vertices[3 * idx.vertex_index + 0],
                        attrib.vertices[3 * idx.vertex_index + 1],
                        attrib.vertices[3 * idx.vertex_index + 2]
                    };
                }
                glm::vec3 normal = glm::normalize(glm::cross(face_vertices[1] - face_vertices[0], face_vertices[2] - face_vertices[0]));

                for (size_t v = 0; v < fv; v++) {
                    tinyobj::index_t idx = shapes[s].mesh.indices[3 * f + v];
                    Vertex vertex{};
                    vertex.Position = face_vertices[v];
                    if (idx.normal_index >= 0) {
                        vertex.Normal = {
                            attrib.normals[3 * idx.normal_index + 0],
                            attrib.normals[3 * idx.normal_index + 1],
                            attrib.normals[3 * idx.normal_index + 2]
                        };
                    }
                    else {
                        vertex.Normal = normal; // Use calculated normal if none provided
                    }
                    if (idx.texcoord_index >= 0) {
                        vertex.TexCoords = {
                            attrib.texcoords[2 * idx.texcoord_index + 0],
                            attrib.texcoords[2 * idx.texcoord_index + 1]
                        };
                    }
                    vertices.push_back(vertex);
                }
            }
        }
        if (!loadTexture(texturePath))
            return false;
        setupMesh();
        return true;
    }
    bool loadTexture(const std::string& path) {
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        int width, height, nrChannels;
        unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrChannels, 0);
        if (data) {
            GLenum format;
            if (nrChannels == 1)
                format = GL_RED;
            else if (nrChannels == 3)
                format = GL_RGB;
            else if (nrChannels == 4)
                format = GL_RGBA;
            else {
                std::cerr << "Unsupported texture format. Number of channels: " << nrChannels << std::endl;
                stbi_image_free(data);
                return false;
            }
            glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);
        }
        else {
            std::cerr << "Failed to load texture: " << stbi_failure_reason() << std::endl;
            return false;
        }
        stbi_image_free(data);
        return true;
    }

    void setupMesh() {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Position));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
    }
};
#pragma once
#include "glad.h"
#include "glfw3.h"
#include <iostream>
#include <vector>
#include <filesystem>
#include <random>
#include <cmath>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "Animation_wm.h"

#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>

#include "filesystem_local.h"
#include "shader_m.h"
#include "camera.h"
#include "animator.h"
#include "model_animation.h"

#pragma comment(lib,"assimp-vc140-mt.lib")
#pragma warning(disable: 4996)

// Global variables to store window state
bool isFullscreen = true;
int windowWidth = 800;
int windowHeight = 600;
GLFWmonitor* primaryMonitor;
const GLFWvidmode* mode;
GLFWwindow* window;
// Camera variables
glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
float deltaTime = 0.0f;
float lastFrame = 0.0f;
float yaw = -90.0f;    // initialized to -90.0 degrees since a yaw of 0.0 results in a direction vector pointing to the right so we initially rotate a bit to the left.
float pitch = 0.0f;
float lastX = 400, lastY = 300;
bool firstMouse = true;
float fov = 45.0f;
std::vector<unsigned int> indices;
std::vector<glm::vec3> vertices;
std::vector<glm::vec3> normals;

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

struct Terrain {
    // Terrain data
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    unsigned int VAO, VBO, EBO;
    unsigned int vertexCount;
    float maxTerrainHeight;

    const char* vertexShaderSource = R"(
        #version 460 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        out vec3 Normal;
        out vec3 FragPos;

        void main() {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
            FragPos = vec3(model * vec4(aPos, 1.0));
            Normal = mat3(transpose(inverse(model))) * aNormal;
        }
    )";

    const char* fragmentShaderSource = R"(
        #version 460 core
        out vec4 FragColor;
        in vec3 Normal;
        in vec3 FragPos;

        uniform vec3 lightPos;
        uniform vec3 viewPos;
        uniform float maxTerrainHeight; //uniform for maximum height

        void main() {
            vec3 lightColor = vec3(1.0);

            // Ambient
            float ambientStrength = 0.3;
            vec3 ambient = ambientStrength * lightColor;

            // Diffuse
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;

            // Specular
            float specularStrength = 0.2;
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrength * spec * lightColor;

            vec3 result = ambient + diffuse + specular;

            // Terrain Coloring based on Height
            float heightPercent = FragPos.y / maxTerrainHeight; // Calculate height percentage
            vec3 baseColor = mix(vec3(0.1, 0.2, 0.0), vec3(0.0, 0.7, 0.0), heightPercent); // Dark green to Green
            baseColor = mix(baseColor, vec3(0.5, 0.4, 0.2), max(0.0f, heightPercent - 0.7)); // Add some brown towards the top
            baseColor = mix(baseColor, vec3(0.8, 0.8, 0.8), max(0.0f, heightPercent - 0.9)); // Add some white towards the very top (snow)


            FragColor = vec4(result * baseColor, 1.0); // Apply lighting to the base color
        }
    )";

    unsigned int shaderProgram;

    Terrain(int width, int height) {
        generateTerrain(width, height);
        setupMesh();
        compileShaders();
    }
    ~Terrain() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
        glDeleteProgram(shaderProgram);
    }

    float interpolate(float a0, float a1, float w) {
        return (a1 - a0) * (3.0f - w * 2.0f) * w * w + a0;
    }

    glm::vec2 randomGradient(int ix, int iy) {
        // using a pseudo-random function for gradient
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> distrib(-1.0f, 1.0f);
        float angle = std::fmod((float)(ix * 31 + iy * 71), 360.0f);
        return glm::vec2(cos(angle), sin(angle));
    }

    float dotGridGradient(int ix, int iy, float x, float y) {
        glm::vec2 gradient = randomGradient(ix, iy);
        float dx = x - ix;
        float dy = y - iy;
        return (dx * gradient.x + dy * gradient.y);
    }

    float perlin(float x, float y) {
        int x0 = (int)floor(x);
        int x1 = x0 + 1;
        int y0 = (int)floor(y);
        int y1 = y0 + 1;

        float sx = x - x0;
        float sy = y - y0;

        float n00 = dotGridGradient(x0, y0, x, y);
        float n10 = dotGridGradient(x1, y0, x, y);
        float n01 = dotGridGradient(x0, y1, x, y);
        float n11 = dotGridGradient(x1, y1, x, y);

        float ix0 = interpolate(n00, n10, sx);
        float ix1 = interpolate(n01, n11, sx);

        return interpolate(ix0, ix1, sy);
    }


    void generateTerrain(int width, int height) {
        vertexCount = width * height;
        vertices.resize(vertexCount * 6);
        indices.resize((width - 1) * (height - 1) * 6);

        float yScale = 10.0f;
        float xzScale = 1.0f;
        float frequency = 0.1f;
        float amplitude = 5.0f;
        int index = 0;

        for (int z = 0; z < height; z++) {
            for (int x = 0; x < width; x++) {
                float y = perlin(x * frequency, z * frequency) * amplitude;
                vertices[index * 6 + 0] = x * xzScale;
                vertices[index * 6 + 1] = y;
                vertices[index * 6 + 2] = z * xzScale;

                vertices[index * 6 + 3] = 0.0f; // Initialize normals to 0
                vertices[index * 6 + 4] = 0.0f;
                vertices[index * 6 + 5] = 0.0f;
                index++;
            }
        }
        float maxH = -10000;
        index = 0;
        for (int z = 0; z < height; z++) {
            for (int x = 0; x < width; x++) {
                maxH = std::max(vertices[index * 6 + 1], maxH);
                index++;
            }
        }
        maxTerrainHeight = maxH;

        index = 0;
        for (int z = 0; z < height - 1; z++) {
            for (int x = 0; x < width - 1; x++) {
                glm::vec3 v0 = glm::vec3(vertices[(z * width + x) * 6 + 0], vertices[(z * width + x) * 6 + 1], vertices[(z * width + x) * 6 + 2]);
                glm::vec3 v1 = glm::vec3(vertices[(z * width + x + 1) * 6 + 0], vertices[(z * width + x + 1) * 6 + 1], vertices[(z * width + x + 1) * 6 + 2]);
                glm::vec3 v2 = glm::vec3(vertices[((z + 1) * width + x) * 6 + 0], vertices[((z + 1) * width + x) * 6 + 1], vertices[((z + 1) * width + x) * 6 + 2]);
                glm::vec3 v3 = glm::vec3(vertices[((z + 1) * width + x + 1) * 6 + 0], vertices[((z + 1) * width + x + 1) * 6 + 1], vertices[((z + 1) * width + x + 1) * 6 + 2]);

                glm::vec3 normal1 = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                glm::vec3 normal2 = glm::normalize(glm::cross(v3 - v1, v2 - v1));

                vertices[(z * width + x) * 6 + 3] += normal1.x;
                vertices[(z * width + x) * 6 + 4] += normal1.y;
                vertices[(z * width + x) * 6 + 5] += normal1.z;

                vertices[(z * width + x + 1) * 6 + 3] += normal1.x;
                vertices[(z * width + x + 1) * 6 + 4] += normal1.y;
                vertices[(z * width + x + 1) * 6 + 5] += normal1.z;

                vertices[((z + 1) * width + x) * 6 + 3] += normal1.x;
                vertices[((z + 1) * width + x) * 6 + 4] += normal1.y;
                vertices[((z + 1) * width + x) * 6 + 5] += normal1.z;

                vertices[(z * width + x) * 6 + 3] += normal2.x;
                vertices[(z * width + x) * 6 + 4] += normal2.y;
                vertices[(z * width + x) * 6 + 5] += normal2.z;

                vertices[(z * width + x + 1) * 6 + 3] += normal2.x;
                vertices[(z * width + x + 1) * 6 + 4] += normal2.y;
                vertices[((z + 1) * width + x + 1) * 6 + 3] += normal2.x;
                vertices[((z + 1) * width + x + 1) * 6 + 4] += normal2.y;
                vertices[((z + 1) * width + x + 1) * 6 + 5] += normal2.z;

                vertices[((z + 1) * width + x) * 6 + 3] += normal2.x;
                vertices[((z + 1) * width + x) * 6 + 4] += normal2.y;
                vertices[((z + 1) * width + x) * 6 + 5] += normal2.z;
            }
        }
        index = 0;
        for (int z = 0; z < height; z++) {
            for (int x = 0; x < width; x++) {
                glm::vec3 normal = glm::vec3(vertices[index * 6 + 3], vertices[index * 6 + 4], vertices[index * 6 + 5]);
                normal = glm::normalize(normal);
                vertices[index * 6 + 3] = normal.x;
                vertices[index * 6 + 4] = normal.y;
                vertices[index * 6 + 5] = normal.z;
                index++;
            }
        }

        index = 0;
        for (int z = 0; z < height - 1; z++) {
            for (int x = 0; x < width - 1; x++) {
                int p0 = z * width + x;
                int p1 = z * width + x + 1;
                int p2 = (z + 1) * width + x;
                int p3 = (z + 1) * width + x + 1;

                indices[index++] = p0;
                indices[index++] = p1;
                indices[index++] = p2;

                indices[index++] = p1;
                indices[index++] = p3;
                indices[index++] = p2;
            }
        }
    }

    void setupMesh() {
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        // Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        // Normal attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);
    }
    void compileShaders() {
        // Vertex shader
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
        glCompileShader(vertexShader);
        checkShaderCompileErrors(vertexShader, "VERTEX");
        // Fragment Shader
        unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
        glCompileShader(fragmentShader);
        checkShaderCompileErrors(fragmentShader, "FRAGMENT");

        // Shader Program
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        checkShaderProgramLinkingErrors(shaderProgram);
        // delete shaders as I already linked it
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
    }
    void checkShaderCompileErrors(unsigned int shader, const std::string& type) {
        int success;
        char infoLog[1024];
        if (type == "VERTEX") {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
        else if (type == "FRAGMENT") {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
    }
    void checkShaderProgramLinkingErrors(unsigned int program) {
        int success;
        char infoLog[1024];
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(program, 1024, NULL, infoLog);
            std::cerr << "ERROR::PROGRAM_LINKING_ERROR of type: PROGRAM\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }

    void Draw(glm::mat4 view, glm::mat4 projection, glm::vec3 lightPos, glm::vec3 viewPos) {
        glUseProgram(shaderProgram);

        glm::mat4 model = glm::mat4(1.0f);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection[0][0]);
        glUniform3fv(glGetUniformLocation(shaderProgram, "lightPos"), 1, &lightPos[0]);
        glUniform3fv(glGetUniformLocation(shaderProgram, "viewPos"), 1, &viewPos[0]);
        glUniform1f(glGetUniformLocation(shaderProgram, "maxTerrainHeight"), maxTerrainHeight);

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        glUseProgram(shaderProgram);

    }
};



struct Skybox {
    unsigned int VAO;
    unsigned int VBO;
    unsigned int textureID;
    unsigned int shaderProgram;

    const char* vertexShaderSource = R"(
        #version 460 core
        layout (location = 0) in vec3 aPos;

        out vec3 TexCoords;

        uniform mat4 view;
        uniform mat4 projection;

        void main() {
            TexCoords = aPos;
            gl_Position = projection * view * vec4(aPos, 1.0);
            gl_Position = gl_Position.xyww; // Important for skybox depth
        }
    )";

    const char* fragmentShaderSource = R"(
        #version 460 core
        out vec4 FragColor;
        in vec3 TexCoords;

        uniform samplerCube skybox;

        void main() {
            FragColor = texture(skybox, TexCoords);
        }
    )";

    Skybox(std::vector<std::string> faces) {
        setupMesh();
        textureID = loadCubemap(faces);
        compileShaders();
    }

    ~Skybox() {
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteTextures(1, &textureID);
        glDeleteProgram(shaderProgram);
    }

    void Draw(glm::mat4 view, glm::mat4 projection) {
        glDepthFunc(GL_LEQUAL);
        glUseProgram(shaderProgram);

        // Removing translation part of the view matrix
        view = glm::mat4(glm::mat3(view));

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection[0][0]);
        glBindVertexArray(VAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
        glDepthFunc(GL_LESS);
    }

private:
    void setupMesh() {
        float skyboxVertices[] = {
            // positions          
            -1.0f,  1.0f, -1.0f,
            -1.0f, -1.0f, -1.0f,
             1.0f, -1.0f, -1.0f,
             1.0f, -1.0f, -1.0f,
             1.0f,  1.0f, -1.0f,
            -1.0f,  1.0f, -1.0f,

            -1.0f, -1.0f,  1.0f,
            -1.0f, -1.0f, -1.0f,
            -1.0f,  1.0f, -1.0f,
            -1.0f,  1.0f, -1.0f,
            -1.0f,  1.0f,  1.0f,
            -1.0f, -1.0f,  1.0f,

             1.0f, -1.0f, -1.0f,
             1.0f, -1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f,  1.0f, -1.0f,
             1.0f, -1.0f, -1.0f,

            -1.0f, -1.0f,  1.0f,
            -1.0f,  1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f, -1.0f,  1.0f,
            -1.0f, -1.0f,  1.0f,

            -1.0f,  1.0f, -1.0f,
             1.0f,  1.0f, -1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
            -1.0f,  1.0f,  1.0f,
            -1.0f,  1.0f, -1.0f,

            -1.0f, -1.0f, -1.0f,
            -1.0f, -1.0f,  1.0f,
             1.0f, -1.0f, -1.0f,
             1.0f, -1.0f, -1.0f,
            -1.0f, -1.0f,  1.0f,
             1.0f, -1.0f,  1.0f
        };
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    }

    unsigned int loadCubemap(std::vector<std::string> faces) {
        unsigned int textureID;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

        int width, height, nrChannels;
        for (unsigned int i = 0; i < faces.size(); i++) {
            unsigned char* data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
            if (data) {
                GLenum format;
                if (nrChannels == 1)
                    format = GL_RED;
                else if (nrChannels == 3)
                    format = GL_RGB;
                else if (nrChannels == 4)
                    format = GL_RGBA;
                glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
                stbi_image_free(data);
            }
            else {
                std::cerr << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
                stbi_image_free(data);
                return 0;
            }
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        return textureID;
    }

    void compileShaders() {
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
        glCompileShader(vertexShader);
        checkShaderCompileErrors(vertexShader, "VERTEX");
        unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
        glCompileShader(fragmentShader);
        checkShaderCompileErrors(fragmentShader, "FRAGMENT");
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        checkShaderProgramLinkingErrors(shaderProgram);
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
    }
    void checkShaderCompileErrors(unsigned int shader, const std::string& type) {
        int success;
        char infoLog[1024];
        if (type == "VERTEX") {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
        else if (type == "FRAGMENT") {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
    }

    void checkShaderProgramLinkingErrors(unsigned int program) {
        int success;
        char infoLog[1024];
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(program, 1024, NULL, infoLog);
            std::cerr << "ERROR::PROGRAM_LINKING_ERROR of type: PROGRAM\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
};


void processInput(GLFWwindow* window) {
    float cameraSpeed = 2.5f * deltaTime;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw += xoffset;
    pitch += yoffset;

    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    fov -= (float)yoffset;
    if (fov < 1.0f)
        fov = 1.0f;
    if (fov > 45.0f)
        fov = 45.0f;
}

int initialize() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW." << std::endl;
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(800, 600, "Toward a Futuristic Emerald Isle - By Tanuj Sood", NULL, NULL);
    if (!window) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Check if the context is current
    if (glfwGetCurrentContext() == window) {
        std::cout << "Context is current for this window." << std::endl;
    }
    else {
        std::cerr << "Context is not current for this window." << std::endl;
    }

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    primaryMonitor = glfwGetPrimaryMonitor();
    mode = glfwGetVideoMode(primaryMonitor);
    glViewport(0, 0, 800, 600);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}


int main() {

    if (initialize() == -1)
    {
        std::cout << "Initialization failed" << std::endl;
    }
    
    Model Building1("resources/building/Residential_Buildings_001.obj");

    // Number of instances per row/column
    unsigned int gridWidth = 3; // Number of instances in x direction
    unsigned int gridHeight = 10; // Number of instances in z direction
    unsigned int numInstances = gridWidth * gridHeight;

    // Spacing between instances
    float spacing = 100.0f;

    std::random_device rd;
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> scaleDist(0.5f, 1.5f); // Scale between 0.5 and 1.5
    std::uniform_real_distribution<> rotationDist(0.0f, 360.0f); // Rotation between 0 and 360 degrees
    //std::uniform_real_distribution<> heightVariation(-0.2f, 0.2f); // Slight height variation

    std::vector<glm::mat4> instanceMatrices(numInstances);
    for (unsigned int y = 0; y < gridHeight; y++) {
        for (unsigned int x = 0; x < gridWidth; x++) {
            glm::mat4 model = glm::mat4(1.0f);

            // Calculating position starting from the bottom-left (0, 0)
            float xPos = x * spacing;
            float zPos = y * spacing;

            // Introduce randomness
            //float scale = scaleDist(gen);
            float scale = 0.7;
            float rotation = rotationDist(gen);
            float heightOffset = 0.0; 

            model = glm::translate(model, glm::vec3(xPos, heightOffset, zPos));
            model = glm::rotate(model, glm::radians(rotation), glm::vec3(0.0f, 1.0f, 0.0f));
            model = glm::scale(model, glm::vec3(scale, scale * (1.0f + heightOffset * 2), scale));

            instanceMatrices[y * gridWidth + x] = model;
        }
    }
 
    // Instance VBO
    unsigned int instanceVBO;
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, numInstances * sizeof(glm::mat4), &instanceMatrices[0], GL_STATIC_DRAW);

    // Set up instance attribute in VAO
    for (int i{}; i < Building1.meshes.size(); i++)
    {
        unsigned int BuildingVao = Building1.meshes[i].VAO;

        glBindVertexArray(BuildingVao);
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)0);
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)(1 * sizeof(glm::vec4)));
        glEnableVertexAttribArray(5);
        glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)(2 * sizeof(glm::vec4)));
        glEnableVertexAttribArray(6);
        glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)(3 * sizeof(glm::vec4)));

        glVertexAttribDivisor(3, 1);
        glVertexAttribDivisor(4, 1);
        glVertexAttribDivisor(5, 1);
        glVertexAttribDivisor(6, 1);

    }

    // Instancing Shader Code
    const char* instancingvertexShaderSource = R"(
        #version 460 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;
        layout (location = 2) in vec2 aTexCoords;
        layout (location = 3) in mat4 instanceMatrix;

        out vec3 Normal;
        out vec3 FragPos;
        out vec2 TexCoords;

        uniform mat4 view;
        uniform mat4 projection;

        void main() {
            gl_Position = projection * view * instanceMatrix * vec4(aPos, 1.0);
            Normal = mat3(transpose(inverse(instanceMatrix))) * aNormal;
            FragPos = vec3(instanceMatrix * vec4(aPos, 1.0));
            TexCoords = aTexCoords;
        }
    )";


    // Shader Code
    const char* defaultvertexShaderSource = R"(
        #version 460 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aNormal;
        layout (location = 2) in vec2 aTexCoords;

        out vec3 Normal;
        out vec3 FragPos;
        out vec2 TexCoords;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        void main() {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
            Normal = mat3(transpose(inverse(model))) * aNormal;
            FragPos = vec3(model * vec4(aPos, 1.0));
            TexCoords = aTexCoords;
        }
    )";

    const char* fragmentShaderSource = R"(
        #version 460 core
        out vec4 FragColor;

        in vec3 Normal;
        in vec3 FragPos;
        in vec2 TexCoords;

        uniform sampler2D ourTexture;

        void main() {
            vec3 lightDir = normalize(vec3(10.0,10.0,10.0)-FragPos);
            float diff = max(dot(Normal,lightDir),0.0);
            vec4 diffuse = vec4(diff * vec3(1.0),1.0);

            
            FragColor = mix(diffuse,texture(ourTexture, TexCoords),0.3);
        }
    )";


    // Shader compilation
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &instancingvertexShaderSource, NULL);   //using isntancing vertex shader here
    glCompileShader(vertexShader);
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        return 1;
    }

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
        return 1;
    }

    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        return 1;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    
    // Grid Shader Code
    const char* gridVertexShaderSource = R"(
        #version 460
     
        out vec3 WorldPos;   
        uniform mat4 gVP = mat4(1.0);
        uniform float gGridSize  = 100.0;
        uniform vec3 gCameraWorldPos;
 
        const vec3 Pos[4] = vec3[4](
            vec3(-1.0, 0.0, -1.0), // bottom left
            vec3( 1.0, 0.0, -1.0), // bottom right
            vec3( 1.0, 0.0,  1.0), // top right
            vec3(-1.0, 0.0,  1.0)  // top left
        );
        const int Indices[6] = int[6](0, 2, 1, 2, 0, 3);
        void main() {
            int Index = Indices[gl_VertexID];
            vec3 vPos3 = Pos[Index] * gGridSize;
            
            vPos3.x += gCameraWorldPos.x;
            vPos3.z += gCameraWorldPos.z;
        
            vec4 vPos4 = vec4(vPos3,1.0);
;
            gl_Position = gVP * vPos4;
            WorldPos = vPos3;
        }
        )";
        
    const char* gridFragmentShaderSource = R"(
    #version 460 core
    in vec3 WorldPos;
    layout(location = 0) out vec4 FragColor;
    uniform float gGridCellSize = 0.025;
    uniform float gGridMinPixelBetweenCells = 2.0;
    uniform vec4 gGridColorThin = vec4(0.5, 0.5, 0.5, 1.0);
    uniform vec4 gGridColorThick = vec4(0.0, 0.0, 0.0, 1.0);

    float log10(float x) {
        return log(x) / log(10.0);
    }

    float satf(float x) {
        return clamp(x, 0.0, 1.0);
    }

    vec2 satv(vec2 x) {
        return clamp(x, vec2(0.0), vec2(1.0));
    }

    void main() {
        vec2 dvx = vec2(dFdx(WorldPos.x), dFdy(WorldPos.x));
        vec2 dvy = vec2(dFdx(WorldPos.z), dFdy(WorldPos.z));
        float lx = length(dvx);
        float ly = length(dvy);
        vec2 dudv = vec2(lx, ly);
        float l = length(dudv);

        float LOD = max(0.0, log10(l * gGridMinPixelBetweenCells / gGridCellSize) + 1.0);
        float GridCellSizeLOD0 = gGridCellSize * pow(10.0, floor(LOD));
        float GridCellSizeLOD1 = GridCellSizeLOD0 * 10.0;
        float GridCellSizeLOD2 = GridCellSizeLOD1 * 10.0;
        dudv *= 4.0;

        vec2 mod_div_dudv = mod(WorldPos.xz, GridCellSizeLOD0) / dudv;
        vec2 Lod0a = max(vec2(0.0), 1.0 - abs(satv(mod_div_dudv) * 2.0 - vec2(1.0))); 
        mod_div_dudv = mod(WorldPos.xz, GridCellSizeLOD1) / dudv;
        vec2 Lod1a = max(vec2(0.0), 1.0 - abs(satv(mod_div_dudv) * 2.0 - vec2(1.0))); 
        mod_div_dudv = mod(WorldPos.xz, GridCellSizeLOD2) / dudv;
        vec2 Lod2a = max(vec2(0.0), 1.0 - abs(satv(mod_div_dudv) * 2.0 - vec2(1.0))); 
        
        float Lod0af = max(Lod0a.x, Lod0a.y);
        float Lod1af = max(Lod1a.x, Lod1a.y);
        float Lod2af = max(Lod2a.x, Lod2a.y);

        float LOD_fade = fract(LOD);
        vec4 Color;
        if (Lod2af > 0.0) {
            Color = gGridColorThick;
            Color.a *= Lod2af;
        } else {
            if (Lod1af > 0.0) {
                Color = mix(gGridColorThick, gGridColorThin, LOD_fade);
                Color.a *= Lod1af;
            } else {
                Color = gGridColorThick;
                Color.a *= (Lod0af * (1.0 - LOD_fade));
            }
        }
        
        float OpacityFallOff = (1.0 - satf(length(WorldPos.xz) / 100.0));
        Color.a *= OpacityFallOff;
        FragColor = Color;
    }
)";

    // Compile Grid Shader
    unsigned int gridVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(gridVertexShader, 1, &gridVertexShaderSource, NULL);
    glCompileShader(gridVertexShader);

    unsigned int gridFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(gridFragmentShader, 1, &gridFragmentShaderSource, NULL);
    glCompileShader(gridFragmentShader);

    unsigned int gridShaderProgram = glCreateProgram();
    glAttachShader(gridShaderProgram, gridVertexShader);
    glAttachShader(gridShaderProgram, gridFragmentShader);
    glLinkProgram(gridShaderProgram);

    glDeleteShader(gridVertexShader);
    glDeleteShader(gridFragmentShader);


    Terrain terrain(400, 400);

    // -------------------------- ANIMATION STUFF--------------------------
    // Shaders
    const char* vertexShaderSourceAnim = R"(
        #version 460 core
        layout (location = 0) in vec3 aPos;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        void main()
        {
           gl_Position = projection * view * model * vec4(aPos, 1.0);
        };
    )";

    const char* fragmentShaderSourceAnim = R"(
        #version 460 core
        out vec4 FragColor;
        void main()
        {
           FragColor = vec4(1.0, 0.5, 0.2, 1.0);
        };
    )";

    // Compile Grid Shader
    unsigned int animVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(animVertexShader, 1, &vertexShaderSourceAnim, NULL);
    glCompileShader(animVertexShader);

    unsigned int animFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(animFragmentShader, 1, &fragmentShaderSourceAnim, NULL);
    glCompileShader(animFragmentShader);

    unsigned int animShaderProgram = glCreateProgram();
    glAttachShader(animShaderProgram, animVertexShader);
    glAttachShader(animShaderProgram, animFragmentShader);
    glLinkProgram(animShaderProgram);

    glDeleteShader(animVertexShader);
    glDeleteShader(animFragmentShader);

    unsigned int VAOAnim, VBOAnim, EBOAnim;
    LoadModel("resources/wind_tower.obj", VAOAnim, VBOAnim, EBOAnim);

    std::vector<std::string> faces{
    "resources/skybox/right.jpg",
    "resources/skybox/left.jpg",
    "resources/skybox/top.jpg",
    "resources/skybox/bottom.jpg",
    "resources/skybox/front.jpg",
    "resources/skybox/back.jpg"
    };

    Skybox skybox(faces); // Create the skybox

    // VAMPIRE MODEL------------------------------------------
    Shader ourShader("anim_model.vs", "anim_model.fs");
    Model ourModel(FileSystem::getPath("resources/vampire/dancing_vampire.dae"));
    Animation danceAnimation(FileSystem::getPath("resources/vampire/dancing_vampire.dae"), &ourModel);
    Animator animator(&danceAnimation);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


    while (!glfwWindowShouldClose(window)) {
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 projection = glm::perspective(glm::radians(fov), (float)windowWidth / (float)windowHeight, 0.1f, 100.0f);

        skybox.Draw(view, projection);

        glm::mat4 VPmatrix = projection * view;
        glUseProgram(gridShaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(gridShaderProgram, "gVP"), 1, GL_FALSE, glm::value_ptr(VPmatrix));
        glUniform3f(glGetUniformLocation(gridShaderProgram, "gCameraWorldPos"), cameraPos.x, cameraPos.y, cameraPos.z);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        terrain.Draw(view, projection, glm::vec3(100.0f), cameraPos);

        // INSTANCED MODEL LOADING FOR BUILDINGS
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniform1i(glGetUniformLocation(shaderProgram, "ourTexture"), 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, Building1.textures_loaded[0].id);

        for (unsigned int i = 0; i < Building1.meshes.size(); i++)
        {
            glBindVertexArray(Building1.meshes[i].VAO);
            glDrawElementsInstanced(GL_TRIANGLES, static_cast<unsigned int>(Building1.meshes[i].indices.size()), GL_UNSIGNED_INT, 0, numInstances);
            glBindVertexArray(0);
        }

        // windmill 1
        glUseProgram(animShaderProgram);
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(20.0, 0.9f, 27.5f));
        model = glm::scale(model, glm::vec3(2, 2, 2));
        model = glm::rotate(model, (float)glfwGetTime(), glm::vec3(0.0f, 1.0f, 0.0f));

        glUniformMatrix4fv(glGetUniformLocation(animShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(animShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(animShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        glBindVertexArray(VAOAnim);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);


        // windmill 2
        glUseProgram(animShaderProgram);
        model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(20.0, 0.9f, 47.5f));
        model = glm::scale(model, glm::vec3(2, 2, 2));
        model = glm::rotate(model, (float)glfwGetTime(), glm::vec3(0.0f, 1.0f, 0.0f));

        glUniformMatrix4fv(glGetUniformLocation(animShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(animShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(animShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        glBindVertexArray(VAOAnim);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        //windmill 3
        glUseProgram(animShaderProgram);
        model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(40.0, 0.9f, 47.5f));
        model = glm::scale(model, glm::vec3(2, 2, 2));
        model = glm::rotate(model, (float)glfwGetTime(), glm::vec3(0.0f, 1.0f, 0.0f));

        glUniformMatrix4fv(glGetUniformLocation(animShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(animShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(animShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        glBindVertexArray(VAOAnim);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        // windmill 4
        glUseProgram(animShaderProgram);
        model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(40.0, 0.9f, 27.5f));
        model = glm::scale(model, glm::vec3(2,2,2));
        model = glm::rotate(model, (float)glfwGetTime(), glm::vec3(0.0f, 1.0f, 0.0f));

        glUniformMatrix4fv(glGetUniformLocation(animShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(animShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(animShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        glBindVertexArray(VAOAnim);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);


        // ---------------------SKELETAL MODEL-----------------------
        animator.UpdateAnimation(deltaTime);

        ourShader.use();
        ourShader.setMat4("projection", projection);
        ourShader.setMat4("view", view);

        auto transforms = animator.GetFinalBoneMatrices();
        for (int i = 0; i < transforms.size(); ++i)
            ourShader.setMat4("finalBonesMatrices[" + std::to_string(i) + "]", transforms[i]);

        model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(100.0f, 9.0f, 100.0f)); 
        //model = glm::scale(model, glm::vec3(.5f, .5f, .5f));	
        ourShader.setMat4("model", model);
        ourModel.Draw(ourShader);


        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glDeleteProgram(shaderProgram);
    glfwTerminate();
    return 0;
}
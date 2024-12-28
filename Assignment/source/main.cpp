#include "glad.h"
#include "glfw3.h"
#include <iostream>
#include <vector>
#include <filesystem>
#include <cmath>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "ObjModelLoader.h"

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

// Global variables to store window state
bool isFullscreen = false;
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

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

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

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(windowWidth, windowHeight, "OpenGL Window", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window." << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD." << std::endl;
        return -1;
    }

    glViewport(0, 0, windowWidth, windowHeight);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    glEnable(GL_DEPTH_TEST);
    return 0;
}

int main() {

    if (initialize() == -1)
    {
        std::cout << "Initialization failed" << std::endl;
    }

    Model Cube("resources/cube.obj", "resources/test_texture.jpg");

    if (Cube.vertices.empty()) {
        std::cerr << "Model loading failed. Exiting." << std::endl;
        return -1;
    }

    Model Teapot("resources/teapot.obj", "resources/test_texture.jpg");

    if (Teapot.vertices.empty()) {
        std::cerr << "Model loading failed. Exiting." << std::endl;
        return -1;
    }

    // Number of instances
    // unsigned int numInstances = 500;


     //std::vector<glm::mat4> instanceMatrices(numInstances);
     //for (unsigned int i = 0; i < numInstances; i++) {
     //    glm::mat4 model = glm::mat4(1.0f);
     //    float x = (rand() % 100 - 50) / 10.0f;
     //    float z = (rand() % 100 - 50) / 10.0f;
     //    model = glm::translate(model, glm::vec3(x, 0.0f, z));
     //    float angle = (rand() % 360);
     //    model = glm::rotate(model, glm::radians(angle), glm::vec3(0.0f, 1.0f, 0.0f));
     //    float scale = (rand() % 10) / 10.0f + 0.5f;
     //    model = glm::scale(model, glm::vec3(scale));
     //    instanceMatrices[i] = model;
     //} 


     // Number of instances per row/column
    unsigned int gridWidth = 100; // Number of instances in x direction
    unsigned int gridHeight = 100; // Number of instances in z direction
    unsigned int numInstances = gridWidth * gridHeight;

    // Spacing between instances
    float spacing = 2.0f;

    std::vector<glm::mat4> instanceMatrices(numInstances);
    for (unsigned int y = 0; y < gridHeight; y++) { // Iterate through rows
        for (unsigned int x = 0; x < gridWidth; x++) { // Iterate through columns
            glm::mat4 model = glm::mat4(1.0f);

            // Calculate position in the grid
            float xPos = (x - (gridWidth - 1) / 2.0f) * spacing; // Center the grid
            float zPos = (y - (gridHeight - 1) / 2.0f) * spacing;
            model = glm::translate(model, glm::vec3(xPos, 0.0f, zPos));

            // Optional: Add some variation (rotation, scale) if needed
            model = glm::rotate(model, glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f)); // Example rotation
            model = glm::scale(model, glm::vec3(0.75f)); // Example uniform scaling

            instanceMatrices[y * gridWidth + x] = model; // Correct index calculation
        }
    }

    // Instance VBO
    unsigned int instanceVBO;
    glGenBuffers(1, &instanceVBO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, numInstances * sizeof(glm::mat4), &instanceMatrices[0], GL_STATIC_DRAW);

    // Set up instance attribute in VAO
    glBindVertexArray(Cube.VAO);
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


    //FragColor = vec4(diffuse, 1.0) * texture(ourTexture, TexCoords);
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

    glUseProgram(shaderProgram);

    glEnable(GL_DEPTH_TEST); // Enable depth testing


    while (!glfwWindowShouldClose(window)) {
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 projection = glm::perspective(glm::radians(fov), (float)windowWidth / (float)windowHeight, 0.1f, 100.0f);

        unsigned int viewLoc = glGetUniformLocation(shaderProgram, "view");
        unsigned int projectionLoc = glGetUniformLocation(shaderProgram, "projection");
        GLint textureLoc = glGetUniformLocation(shaderProgram, "ourTexture");

        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
        glUniform1i(textureLoc, 0);

        glUseProgram(shaderProgram);

        // Draw instances
        glBindVertexArray(Cube.VAO);
        glDrawArraysInstanced(GL_TRIANGLES, 0, Cube.vertices.size(), numInstances); // Instanced drawing

        glfwSwapBuffers(window);
        glfwPollEvents();
    }


    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}


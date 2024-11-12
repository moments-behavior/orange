#include "fs_utils.hpp"
#include <pwd.h>
#include <unistd.h>
#include <iostream>
#include <sys/stat.h>
#include <vector>

namespace fs_utils {

std::filesystem::path get_home_directory() {
    const char* home = getenv("HOME");
    if (!home) {
        struct passwd* pwd = getpwuid(getuid());
        if (pwd)
            home = pwd->pw_dir;
    }
    if (!home)
        throw std::runtime_error("Could not determine home directory");
    return std::filesystem::path(home);
}

std::string get_real_user() {
    // Check for SUDO_USER first (when running under sudo)
    const char* sudo_user = getenv("SUDO_USER");
    if (sudo_user) return sudo_user;
    
    // Otherwise use normal USER env var
    const char* user = getenv("USER");
    if (user) return user;
    
    // Fallback to looking up current UID
    struct passwd* pw = getpwuid(getuid());
    if (pw) return pw->pw_name;
    
    throw std::runtime_error("Could not determine real user");
}

bool initialize_directories() {
    try {
        // Get real user info first
        std::string username = get_real_user();
        uid_t uid;
        gid_t gid;
        if (!get_user_ids(username, uid, gid)) {
            std::cerr << "Failed to get user IDs for " << username << std::endl;
            return false;
        }

        std::vector<std::string> required_dirs = {
            "recordings",
            "config",
            "config/local",
            "config/network",
            "exp",
            "exp/unsorted",
            "detect",
            "pictures"
        };

        // Get home directory and create base orange_data directory
        std::filesystem::path base_dir = get_home_directory() / "orange_data";
        
        // Create the base directory first
        if (!std::filesystem::exists(base_dir)) {
            if (!std::filesystem::create_directory(base_dir)) {
                std::cerr << "Failed to create base directory: " << base_dir << std::endl;
                return false;
            }
            std::cout << "Created base directory: " << base_dir << std::endl;
        }
        
        // Set ownership of base directory
        if (!set_ownership_and_perms(base_dir.string(), uid, gid, DIR_PERMS)) {
            std::cerr << "Failed to set ownership of base directory" << std::endl;
            return false;
        }
        
        // Create each required subdirectory
        for (const auto& dir : required_dirs) {
            std::filesystem::path dir_path = base_dir / dir;
            if (!std::filesystem::exists(dir_path)) {
                if (std::filesystem::create_directories(dir_path)) {
                    std::cout << "Created directory: " << dir_path << std::endl;
                } else {
                    std::cerr << "Failed to create directory: " << dir_path << std::endl;
                    return false;
                }
            }
            // Set ownership of each directory
            if (!set_ownership_and_perms(dir_path.string(), uid, gid, DIR_PERMS)) {
                std::cerr << "Failed to set ownership of: " << dir_path << std::endl;
                return false;
            }
        }

        return true;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing directories: " << e.what() << std::endl;
        return false;
    }
}

bool get_user_ids(const std::string& username, uid_t& uid, gid_t& gid) {
    struct passwd *pw = getpwnam(username.c_str());
    if (pw == nullptr) {
        std::cerr << "Error: Could not get user info for " << username << std::endl;
        return false;
    }
    uid = pw->pw_uid;
    gid = pw->pw_gid;
    return true;
}

bool set_ownership_and_perms(const std::string& path, uid_t uid, gid_t gid, mode_t perms) {
    bool success = true;
    if (chmod(path.c_str(), perms) != 0) {
        std::cerr << "Error: Could not set permissions for " << path << std::endl;
        success = false;
    }
    if (chown(path.c_str(), uid, gid) != 0) {
        std::cerr << "Error: Could not set ownership for " << path << std::endl;
        success = false;
    }
    return success;
}

bool create_dir_with_ownership(const std::string& path, uid_t uid, gid_t gid, mode_t perms) {
    try {
        if (!std::filesystem::exists(path)) {
            std::filesystem::create_directories(path);
            std::cout << "Created directory: " << path << std::endl;
        }
        
        // Set ownership and permissions for the target directory and all its parents under orange_data
        std::filesystem::path current_path(path);
        while (current_path.string().find("orange_data") != std::string::npos) {
            if (std::filesystem::exists(current_path)) {
                if (!set_ownership_and_perms(current_path.string(), uid, gid, perms)) {
                    std::cerr << "Failed to set ownership/permissions for: " << current_path << std::endl;
                    return false;
                }
            }
            current_path = current_path.parent_path();
        }
        return true;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error creating directory " << path << ": " << e.what() << std::endl;
        return false;
    }
}

bool set_recursive_ownership(const std::string& path, uid_t uid, gid_t gid,
                           mode_t dir_perms, mode_t file_perms) {
    bool success = true;
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
            mode_t perms = entry.is_directory() ? dir_perms : file_perms;
            if (!set_ownership_and_perms(entry.path().string(), uid, gid, perms)) {
                success = false;
            }
        }
        // Don't forget the root directory itself
        if (!set_ownership_and_perms(path, uid, gid, dir_perms)) {
            success = false;
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error during recursive permission setting: " << e.what() << std::endl;
        success = false;
    }
    return success;
}

} // namespace fs_utils
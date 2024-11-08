#include "fs_utils.hpp"
#include <pwd.h>
#include <unistd.h>
#include <iostream>
#include <sys/stat.h>
#include <vector>

namespace fs_utils {

bool initialize_directories() {
    try {
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

        // Get current working directory or specify base directory
        std::filesystem::path base_dir = std::filesystem::current_path();
        
        // Create each required directory
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
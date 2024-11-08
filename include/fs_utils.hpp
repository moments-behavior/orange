#pragma once

#include <string>
#include <filesystem>
#include <sys/types.h>
#include <sys/stat.h>

namespace fs_utils {

bool initialize_directories();

// Standard permission constants
constexpr mode_t DIR_PERMS = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;  // 755
constexpr mode_t FILE_PERMS = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;           // 644

/**
 * @brief Gets the UID and GID for a given username
 * @param username The username to look up
 * @param uid Output parameter for user ID
 * @param gid Output parameter for group ID
 * @return true if successful, false otherwise
 */
bool get_user_ids(const std::string& username, uid_t& uid, gid_t& gid);

/**
 * @brief Sets ownership and permissions for a file or directory
 * @param path Path to the file or directory
 * @param uid User ID to set as owner
 * @param gid Group ID to set as group
 * @param perms Permissions to set
 * @return true if successful, false otherwise
 */
bool set_ownership_and_perms(const std::string& path, uid_t uid, gid_t gid, mode_t perms);

/**
 * @brief Creates a directory with specified ownership and permissions
 * @param path Path to create
 * @param uid User ID to set as owner
 * @param gid Group ID to set as group
 * @param perms Permissions to set (defaults to DIR_PERMS)
 * @return true if successful, false otherwise
 */
bool create_dir_with_ownership(const std::string& path, uid_t uid, gid_t gid, 
                             mode_t perms = DIR_PERMS);

/**
 * @brief Recursively sets ownership and permissions for a directory and its contents
 * @param path Root directory path
 * @param uid User ID to set as owner
 * @param gid Group ID to set as group
 * @param dir_perms Permissions for directories (defaults to DIR_PERMS)
 * @param file_perms Permissions for files (defaults to FILE_PERMS)
 * @return true if successful, false otherwise
 */
bool set_recursive_ownership(const std::string& path, uid_t uid, gid_t gid,
                           mode_t dir_perms = DIR_PERMS,
                           mode_t file_perms = FILE_PERMS);

} // namespace fs_utils
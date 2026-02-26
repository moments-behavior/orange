# Recording modes and where video is saved

## Local vs network mode (where they are in the UI)

- **Local mode**  
  - **Window:** ImGui window titled **"Local"** in `orange.cpp` (around line 951).  
  - **Config:** Camera configs are loaded from `orange_data/config/local/` (e.g. subfolders like `5cam`, `center_ceiling`). You pick one via the "Local" radio buttons.  
  - **Flow:** Click **"Open camera"** (with a local config selected) → cameras open on this machine → click the **play (▶)** button to start recording.  
  - **Code:** Local branch uses `local_config_folders[local_config_select]`, then `start_camera_streaming(..., encoder_config->folder_name, ...)` with `encoder_config->folder_name = input_folder + "/" + get_current_date_time()`.

- **Network mode**  
  - **Window:** The **network** panel in `orange.cpp` (around lines 205–375) where you see "Open Cameras", client status, and **"Clients start camera threads"** / **"Start Recording"**.  
  - **Config:** Camera configs are loaded from `orange_data/config/network/` (e.g. `rig_new`). You pick one via the network radio buttons.  
  - **Flow:** Click **"Open Cameras"** → server sends `OPENCAMERA` to clients (dosa0, dosa1) with the selected config folder → click **"Clients start camera threads"** (or **"Start Recording"**) → server sends `STARTTHREAD` with `record_folder` and encoder setup; each client creates that folder locally and starts encoding.  
  - **Code:** Network branch uses `network_config_folders[network_config_select]`, `host_broadcast_open_cameras`, then `host_broadcast_start_threads(fb_builder, &server, encoder_config->folder_name, encoder_setup)`.

So: **local** = "Local" window + "Open camera" + play. **Network** = network panel + "Open Cameras" + "Clients start camera threads" / "Start Recording".

---

## Where video is saved

- **Base path (both modes)**  
  - Built in `orange.cpp` as:
    - `recording_root_dir_str = "/home/" + tokenized_path[2] + "/orange_data"`
    - `input_folder = recording_root_dir_str + "/exp/unsorted"`
  - So the root is `~/orange_data` (user = 3rd path component of current working directory).  
  - You can change the directory via the **"Save to"** button (sets `input_folder` from the file dialog).

- **Local mode**  
  - Folder for a run: `input_folder + "/" + get_current_date_time()`  
    → e.g. `~/orange_data/exp/unsorted/2025-02-24_12-30-00`.  
  - Video files: **on the same machine** running the orange GUI:
    - `Cam<camera_serial>.mp4`
    - `Cam<camera_serial>_meta.csv`
    - `Cam<camera_serial>_keyframe.csv`  
  - Implemented in `video_capture.cpp` (`acquire_frames` → `GPUVideoEncoder` with `folder_name`) and `gpu_video_encoder.cpp` (`initialize_writer`).

- **Network mode**  
  - The **server** builds the same path: `input_folder + "/" + get_current_date_time()` and sends it in `STARTTHREAD` as `record_folder`.  
  - **Each client** (dosa0, dosa1) receives that path and:
    - Creates it with `mkdir(record_folder, 0777)` in `orange_headless_client.cpp` (`start_camera_thread`).
    - Passes it to `acquire_frames(..., folder_name, ...)` → `GPUVideoEncoder`.  
  - So in network mode, video is saved **on each client machine** (dosa0, dosa1) under the **exact path string the server sent** (e.g. `/home/<server_user>/orange_data/exp/unsorted/<datetime>/`).

---

## Why video might not be saved

1. **Network mode: path uses server’s home (wrong on client)**  
   - The server builds the path using its own current working directory: `"/home/" + tokenized_path[2] + "/orange_data/exp/unsorted/..."`. So it contains the **server’s** username.  
   - Clients use that string as-is. On dosa0/dosa1 that might be e.g. `/home/serveruser/orange_data/...`. If that directory doesn’t exist or isn’t writable on the client (e.g. different user, no shared home), **mkdir or file creation can fail** and video won’t be saved.  
   - **Fix:** Either run clients with the same username and same home layout, or make the recording path configurable per client (e.g. client-side `orange_data` root) so the client builds a path valid on its own machine.

2. **Encoder never created**  
   - `GPUVideoEncoder` is only created when **both** `camera_control->record_video` and `camera_select->record` are true (`video_capture.cpp` around 276–278).  
   - In network mode, `record_video` is set when handling `STARTTHREAD`; per-camera `record` is true by default. If something clears `record` or the client never reaches the branch that sets `record_video`, no encoder runs and no video is written.

3. **Frames not reaching the encoder**  
   - Frames are pushed in `get_one_frame()` when `camera_control->record_video && camera_select->record` (around 180–185 in `video_capture.cpp`). If the capture loop exits early (e.g. stop, error, or CUDA/illegal memory access), the encoder thread may get no frames or stop before writing.

4. **CUDA / GPU errors**  
   - You’ve seen “illegal memory access” and “Failed to copy frame to CPU” in `FrameSaver` / `obb_detector`. Similar errors in the encoder path (e.g. in `gpu_video_encoder.cpp` or capture) can stop the recording thread and prevent finalizing the MP4.

5. **Local mode: “Save to” or folder creation**  
   - If `input_folder` was changed to a read-only or invalid path, or `make_folder(encoder_config->folder_name)` fails, the encoder could fail to open the output file. Check that the path exists and is writable after clicking the play button.

6. **Client `mkdir` failure**  
   - In `start_camera_thread` (`orange_headless_client.cpp`), if `mkdir(record_folder.c_str(), 0777)` fails (e.g. path invalid on client), the function returns false and the camera thread is not started, so no recording on that client. The client logs “Error : …” and “Recorded video saves to : …” only on success.

---

## Quick checks when video is not saved

- **Local:** Confirm you used the **play (▶)** button after “Open camera”, and check `~/orange_data/exp/unsorted/<datetime>/` for `Cam<serial>.mp4`. Ensure “record” is checked for the camera(s) in the table.  
- **Network:** On each client, check the log for “Recorded video saves to : …” and any “Error :” from `mkdir`. Then check that same path on the **client** filesystem for `Cam<serial>.mp4`. If the path contains the server’s username and the client has a different user, create the path on the client (e.g. `~/orange_data/exp/unsorted/<datetime>`) or change the design so the client uses its own base path.

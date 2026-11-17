{
  description = "ROS 2 Jazzy and KITTI metric-depth evaluation on AMD Radeon";

  inputs = {
    nix-ros-overlay.url = "github:lopsided98/nix-ros-overlay/master";

    # Follow the Nixpkgs revision used by the ROS overlay.
    nixpkgs.follows = "nix-ros-overlay/nixpkgs";
  };

  outputs = { nixpkgs, nix-ros-overlay, ... }:
    let
      system = "x86_64-linux";

      # Keep PyTorch and its dependencies on the same ROCm-enabled package set.
      pkgs = import nixpkgs {
        inherit system;

        overlays = [
          nix-ros-overlay.overlays.default
        ];

        config = {
          allowUnfree = true;
          cudaSupport = false;
          rocmSupport = true;
        };
      };

      ros = pkgs.rosPackages.jazzy;

      # Use the Python interpreter and package set associated with ROS.
      python = ros.python3.withPackages (
        _: with ros.python3Packages; [
          # Model inference and checkpoint loading.
          torch
          torchvision
          transformers
          huggingface-hub
          safetensors

          # Image handling, geometry, metrics, and plotting.
          numpy
          scipy
          pillow
          opencv4
          matplotlib

          # Supporting utilities.
          requests
          pyyaml
          tqdm
          pip
        ]
      );

      rosEnv = ros.buildEnv {
        underlay = true;

        paths = with ros; [
          ros-base
          rclpy
          sensor-msgs
          sensor-msgs-py
          diagnostic-msgs
          cv-bridge
          tf2-ros
          rosbag2
          demo-nodes-py
        ];
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "kitti-depth-ros2";

        packages = [
          python
          rosEnv
          pkgs.colcon
          pkgs.cmake
          pkgs.pkg-config
          pkgs.git
          pkgs.ruff
          pkgs.rocmPackages.rocminfo
        ];

        # Prevent user-installed Python packages from affecting imports.
        PYTHONNOUSERSITE = "1";

        # Allow plots to be saved without a graphical session.
        MPLBACKEND = "Agg";
        
        #Uses project's python
        shellHook = ''
          export PATH="${python}/bin:$PATH"
        '';
      };
    };
}
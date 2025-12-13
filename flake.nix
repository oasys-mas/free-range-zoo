{
  inputs = {
    nixpkgs.url = "github:cachix/devenv-nixpkgs/rolling";
    devenv = {
      url = "github:cachix/devenv";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    flake-utils.url = "github:numtide/flake-utils";
    treefmt-nix.url = "github:numtide/treefmt-nix";

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };
  outputs = {
    nixpkgs,
    devenv,
    flake-utils,
    treefmt-nix,
    ...
  } @ inputs:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        treefmtConfig = {...}: {
          projectRootFile = "flake.nix";
          programs = {
            alejandra.enable = true;
            yapf.enable = true;
          };
        };

        treefmtEval = treefmt-nix.lib.evalModule pkgs (treefmtConfig {inherit pkgs;});
      in {
        devShells = {
          ci = pkgs.mkShell {
            nativeBuildInputs = with pkgs; [
              pkg-config
              cargo
              rustc

              python312
            ];

            packages = with pkgs; [
              alejandra
              yapf

              poetry

              python312Packages.virtualenv
            ];

            LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
          };

          default = devenv.lib.mkShell {
            inherit inputs pkgs;

            modules = [
              ({pkgs, ...}: {
                languages = {
                  nix.enable = true;
                  python = {
                    enable = true;
                    package = pkgs.python312;
                    poetry = {
                      enable = true;
                      activate.enable = true;
                      install.enable = true;
                    };
                  };
                  rust = {
                    enable = true;
                    channel = "stable";
                  };
                };

                packages = with pkgs; [
                  alejandra
                  bacon
                  cargo-release
                  clippy
                  presenterm
                  rustfmt
                  rusty-man
                  yapf
                ];

                tasks = {
                  "frz:test" = {
                    exec = "python -m unittest -b";
                    showOutput = true;
                  };
                };

                enterShell = ''
                  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NIX_LD_LIBRARY_PATH
                  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/run/opengl-driver/lib

                  export CUDA_ROOT="${pkgs.cudaPackages.cudatoolkit}"
                '';
              })
            ];
          };
        };

        formatter = treefmtEval.config.build.wrapper;
      }
    );
}

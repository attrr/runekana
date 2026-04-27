{
  description = "Furigana annotation tool for Japanese EPUB files";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.callPackage ./default.nix { };
        }
      );

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/runekana";
          inherit (self.packages.${system}.default) meta;
        };
      });

      checks = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pkg = self.packages.${system}.default;
          # Create a python environment with all dependencies for testing
          pythonWithDeps = pkgs.python3.withPackages (
            ps:
            pkg.propagatedBuildInputs
            ++ [
              ps.pytest
              ps.types-lxml
              ps.types-requests
            ]
          );
        in
        {
          ruff = pkgs.runCommand "ruff-check" { nativeBuildInputs = [ pkgs.ruff ]; } ''
            ruff check ${./.}
            touch $out
          '';

          pyright =
            pkgs.runCommand "pyright-check"
              {
                nativeBuildInputs = [
                  pkgs.pyright
                  pythonWithDeps
                ];
              }
              ''
                export HOME=$TMPDIR
                export PYTHONPATH=${./.}/src
                pyright --project ${./.}/pyproject.toml ${./.}
                touch $out
              '';

          pytest = pkgs.runCommand "pytest-check" { nativeBuildInputs = [ pythonWithDeps ]; } ''
            export HOME=$TMPDIR
            export PYTHONPATH=${./.}/src
            # -p no:cacheprovider avoids Permission Denied warnings in the Nix sandbox
            pytest -p no:cacheprovider ${./.}/tests
            touch $out
          '';

          epubcheck =
            pkgs.runCommand "output-epubcheck"
              {
                nativeBuildInputs = [
                  pkg
                  pkgs.epubcheck
                ];
              }
              ''
                export HOME=$TMPDIR
                runekana ${./.}/tests/fixtures/rashomon.epub out.epub
                epubcheck out.epub
                touch $out
              '';
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            inputsFrom = [ self.packages.${system}.default ];
            packages = with pkgs; [
              ruff
              black
              pyright
              epubcheck
              sudachi-rs
              python3Packages.pytest
              python3Packages.coverage
              # types
              python3Packages.types-lxml
              python3Packages.types-requests
            ];
          };
        }
      );
    };
}

{
  description = "Qwen Flash trellis quantisation and PLE-SSD serving";
  inputs.nixpkgs.url = "https://releases.nixos.org/nixos/unstable/nixos-26.11pre1103272.31ecbaca1bde/nixexprs.tar.zst";
  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      runtime = import ./nix/runtime.nix { inherit pkgs; };
    in {
      packages.${system} = {
        default = runtime.package;
        inherit (runtime) kernels overlay;
      };
      apps.${system}.default = {
        type = "app";
        program = "${runtime.package}/bin/t3";
      };
      devShells.${system}.default = pkgs.mkShell {
        packages = [ runtime.package pkgs.git pkgs.python312 pkgs.nixfmt ];
      };
      nixosModules.default = import ./nix/module.nix self;
    };
}

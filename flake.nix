# src: https://tonyfinn.com/blog/nix-from-first-principles-flake-edition/nix-8-flakes-and-developer-environments
{
	inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

	outputs = { self, nixpkgs }:
	let
		pkgs = import nixpkgs { system = "x86_64-linux"; };
	in {
		devShells.x86_64-linux.default = pkgs.mkShell {
			packages = with pkgs; [
				(pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
					torch # pytorch
					torchvision
					# Add other python dependencies here:
				]))
				# Add other dependencies here:
			];
			# Set environment variables here:
			# MY_ENV_VAR = 1;
		};
		# Define extra shells or packages here.
	};
}

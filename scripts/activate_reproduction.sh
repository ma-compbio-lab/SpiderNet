# Source from the repository root: source scripts/activate_reproduction.sh
_spidernet_activate_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_spidernet_env="$(python -B "$_spidernet_activate_dir/release_environment.py" --configure --shell bash)" || return 1
eval "$_spidernet_env"
unset _spidernet_activate_dir _spidernet_env

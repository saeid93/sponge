# Uncommon steps from Chameleon
1. Install [Nvidia Driver](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) 
    1. Don't forget to do [this step](https://askubuntu.com/questions/841876/how-to-disable-nouveau-kernel-driver)
2. There is a known bug of [snap](https://forum.snapcraft.io/t/permission-denied-on-launch/909) when working with none-standard home directories, this will stop microk8s working, a simple workaournd is to add the following line to `~/.zshrc`.
```bash
alias microk8s="sudo microk8s"
```
3. For the same problem of irregular directory of homes, this should be added to the `~/.zshrc` for having the conda packages
```
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/users/saeid93/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/users/saeid93/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/users/saeid93/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/users/saeid93/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

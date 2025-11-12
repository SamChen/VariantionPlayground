## Install editor when we need it.
## Free docker build from consider editors and editors' environment.

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
test -d ~/.linuxbrew && eval "$(~/.linuxbrew/bin/brew shellenv)"
test -d /home/linuxbrew/.linuxbrew && eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> ~/.zshrc
brew install nodejs universal-ctags
brew install vim

cd /home/envd/
git clone https://github.com/SamChen/vim_config.git
ln -s /home/envd/vim_config/init.vim /home/envd/.vimrc
ln -s /home/envd/vim_config /home/envd/.vim
cd /home/envd/vim_config/
git checkout coc_config

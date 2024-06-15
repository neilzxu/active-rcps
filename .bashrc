# ----- guard against non-interactive logins ---------------------------------
[ -z "$PS1" ] && return


# Load last directory terminal was open
if [ -e ~/.last_cd ]
then
    cd $(cat ~/.last_cd)
fi

# ----- convenient alias and function definitions ----------------------------

# color support for ls and grep
alias grep='grep --color=auto'
if [[ `uname` = "Darwin" || `uname` = "FreeBSD" ]]; then
  alias ls='ls -G'
else
  alias ls='ls --color=auto'
fi

alias killz='killall -9 '
alias hidden='ls -a | grep "^\..*"'
alias rm='rm -v'
alias cp='cp -v'
alias mv='mv -v'
alias shell='ps -p $$ -o comm='


logged_cd() {
    \cd "$@"
    pwd > ~/.last_cd
    ls
}
alias cd="logged_cd"

# ----- shell settings and completion -------------------------------------

# Make .bash_history store more and not store duplicates
export HISTCONTROL=ignoreboth
export HISTSIZE=250000
export HISTFILESIZE=250000

# Append to the history file, dont overwrite it
shopt -s histappend

# Check the window size after each command and, if necessary,
# Update the values of LINES and COLUMNS.
shopt -s checkwinsize

# Enable programmable completion features

if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

bind "set completion-ignore-case on"

# Make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe.sh ] && export LESSOPEN="|/usr/bin/lesspipe.sh %s"

# Turn off the ability for other people to message your terminal using wall
#mesg n


# ----- change the prompt ----------------------------------------------------

# It's really fun to customize your prompt.
# Give it a try! See man bash for help


txtblk='\e[0;30m' # Black - Regular
txtred='\e[0;31m' # Red
txtgrn='\e[0;32m' # Green
txtylw='\e[0;33m' # Yellow
txtblu='\e[0;34m' # Blue
txtpur='\e[0;35m' # Purple
txtcyn='\e[0;36m' # Cyan
txtwht='\e[0;37m' # White
bldblk='\e[1;30m' # Black - Bold
bldred='\e[1;31m' # Red
bldgrn='\e[1;32m' # Green
bldylw='\e[1;33m' # Yellow
bldblu='\e[1;34m' # Blue
bldpur='\e[1;35m' # Purple
bldcyn='\e[1;36m' # Cyan
bldwht='\e[1;37m' # White
unkblk='\e[4;30m' # Black - Underline
undred='\e[4;31m' # Red
undgrn='\e[4;32m' # Green
undylw='\e[4;33m' # Yellow
undblu='\e[4;34m' # Blue
undpur='\e[4;35m' # Purple
undcyn='\e[4;36m' # Cyan
undwht='\e[4;37m' # White
bakblk='\e[40m'   # Black - Background
bakred='\e[41m'   # Red
badgrn='\e[42m'   # Green
bakylw='\e[43m'   # Yellow
bakblu='\e[44m'   # Blue
bakpur='\e[45m'   # Purple
bakcyn='\e[46m'   # Cyan
bakwht='\e[47m'   # White
txtrst='\e[0m'    # Text Reset


#PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '

# determine git branch name
function parse_git_branch(){
  git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}

# determine mercurial branch name
function parse_hg_branch(){
  hg branch 2> /dev/null | awk '{print " (" $1 ")"}'
}

# Determine the branch/state information for this git repository.
function set_git_branch() {
  # Get the name of the branch.
  branch=""
  if [ -x "$(command -v git)" ]; then
    branch=$(parse_git_branch)
  fi

  # if not git then maybe mercurial
  if [ -x "$(command -v hg)" ]; then
    branch=$(parse_hg_branch)
  fi

  # Set the final branch string.
  if [[ "$branch" != "" ]]; then
    BRANCH="${bldpur}${branch}${txtrst} "
  else
    BRANCH=""
  fi
}

# Determine active Python virtualenv details.
function set_virtualenv () {
  if test -z "$VIRTUAL_ENV" ; then
      PYTHON_VIRTUALENV=""
  else
      PYTHON_VIRTUALENV="${txtcyn}[`basename \"$VIRTUAL_ENV\"`]${txtrst} "
  fi
}

# Determine active Python conda environment details.
function set_condaenv () {
  if test -z "$CONDA_DEFAULT_ENV" ; then
      PYTHON_CONDAENV=""
  else
      PYTHON_CONDAENV=" ${txtcyn}[`basename \"$CONDA_DEFAULT_ENV\"`]${txtrst} "
  fi
}


prompt_time="\e[0;20m"
prompt_dow="\e[0;02m"
set_prompt(){
    local last_command=$?
    # set PYTHON_VIRTUALENV
    set_virtualenv
    # set PYTHON_CONDAENV
    set_condaenv
    # set BRANCH
    set_git_branch
    PS1="\[${bakblk}\]\n\[${txtcyn}\]\[${prompt_dow}\]\d \D{%Y} :: \[${prompt_time}\]\t\[${txtylw}\] |-> \w${PYTHON_CONDAENV}${PYTHON_VIRTUALENV}${BRANCH} \n"
    if [[ $last_command != 0 ]]; then
        PS1+="\[$txtred\]\u $ "
    else
        PS1+="\[$txtgrn\]\u $ "
    fi
    PS1+='\[\e[0m\]'
}
# Set 'ls' directory color to be brighter
LS_COLORS=$LS_COLORS:'di=0;94:' ; export LS_COLORS

PROMPT_COMMAND='set_prompt'

function update_bashrc() {
    # Install fzf
    if [ ! -d "$HOME/.fzf" ]; then
        git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
        $HOME/.fzf/install

        source /usr/share/fzf/key-bindings.bash
        source /usr/share/fzf/completion.bash
    fi
    # Update dotfiles
    if [[ -d $HOME/.dotfiles ]]; then
        git --git-dir=$HOME/.dotfiles/.git pull origin master
    fi
}

# Find best finder for fzf

SEARCH_BINS=("ag" "rg" "fd")
SEARCH_CMDS=("ag -l --nocolor -g \"\""
"rg --files --follow"
"fd")

for idx in $(seq 0 $((${#SEARCH_BINS[@]} - 1))); do
    findbin=${SEARCH_BINS[$idx]}
    if [[ $(which $findbin 2> /dev/null) != "" ]]; then
        export FZF_DEFAULT_COMMAND="${SEARCH_CMDS[$idx]}"
        echo "Found alternative to find for fzf: ${FZF_DEFAULT_COMMAND}"
        break
    fi
done

# Ensure git editor uses vim
export GIT_EDITOR=vim

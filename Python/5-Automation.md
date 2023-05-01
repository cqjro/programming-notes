# Automation using Python

## Creating an Executable
```python

# THIS IS ALL WRITTEN IN THE TERMINAL

cd # 'change directory' -> switch the directory containing the py file
pyinstaller --onfile '*insert py file name*' # creates the executable for the py file
```

## Automating Time when Excutables Run
```python
# THIS IS ALL WRITTEN IN THE TERMINAL



#Step 1: Open crontab
crontab -e 

#Step 2: Build time automated command, using crontab.guru to help build the automation
'*paste command into terminal*'

# Step 3: press 'i' key on keyboard to go into insert both

# Step 4: drag exec into terminal to paste the directory to exec

# Step 5: press esc to exit insert mode

# Step 6: save the commaned
:wq # stands for write and quit

# Step 7: verify that the command was created
crontab -l 
```

## Saving Files to the same directory as executable
```python
from datetime import dateime
import os
import sys

# Step 1: Get the path of the executable that we create for this .py file
application_path = os.path.dirname(sys.executable)

# Step 2: Get the date and time (only if creating new file and you want to include the time it was run)
now = datetime.now()
time_format = now.strtime() # get the formating for this time at strtime.org

# Step 3:
```

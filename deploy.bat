@echo off
setlocal

echo ===================================================
echo =      Discloud & Git Deployment Assistant      =
echo ===================================================
echo.

REM --- Check for Python ---
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python and add it to your PATH to run this script.
    goto end
)

REM --- Run the Python helper script for deployment ---
echo [STEP 1] Running Python script to handle deployment to Discloud...
python deploy_helper.py

REM --- Check the result from the Python script ---
REM The python script will exit with code 0 on success, and non-zero on failure.
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The Python script failed. See the output above for details.
    echo Aborting Git operations.
    goto end
)

echo.
echo [STEP 2] Discloud deployment successful and app is confirmed online.
echo Checking for local changes to commit...
echo.

REM --- Check if there are any changes to commit ---
set "changes_found="
for /f "tokens=*" %%a in ('git status --porcelain') do (
    set "changes_found=true"
)

if not defined changes_found (
    echo No local changes to commit. All done!
    goto end
)

echo Found changes to commit.
echo.

REM --- Ask for commit message and perform Git operations ---
:get_commit_message
set "commit_msg="
set /p commit_msg="Enter your commit message: "

REM Check if the commit message is empty
if not defined commit_msg (
    echo Commit message cannot be empty. Please try again.
    goto get_commit_message
)

echo.
echo [STEP 3] Adding, committing, and pushing changes to remote repository...
echo.

git add -A
git commit -m "%commit_msg%"
git push origin

if %errorlevel% neq 0 (
    echo [ERROR] Git push failed. Please check your connection and credentials.
) else (
    echo.
    echo Successfully pushed changes to origin.
)

:end
echo.
echo ===================================================
echo Script finished.
echo ===================================================
pause
endlocal
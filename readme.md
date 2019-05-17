# MD

## Made by

    Mark Jervelund          (mjerv15)
    Morten Kristian Jæger   (mojae15)

## required libs
   
   Keres
   Pandas
   Tensorflow



## Git

How to create branch of old commit

```bash
git branch <branchname> <hash value of commit>
```

How to switch to branch 

```bash
git checkout <branchname>
```

How to delete branch

```bash
git branch -d <branchname>
```

How to list branches

```bash
git branch
```

Show previous commits in terminal

```bash
git log --stat
```

Go back n commits


```bash
git checkout HEAD-n
```

Git Bisectb

```bash
git bisect start

# Current head / commit
git bisect bad

# Last known good coomit
git bisect good <commit hash>

---->
# Tell bisect if the commit is good or bad.
git bisect good/bad

# If any changes is made during testing
git reset --hard HEAD
# - to be able to move on to another commit

---->
# Create branch from bad commit or reset.
git bisect reset / git branch <branchname> 
```
# Contribute to *OpenR*

> Every contribution, no matter how small, is valuable to the community.

Thank you for your interest in ***OpenR*** ! 🥰 We are deeply committed to the open-source community, 
and we welcome contributions from everyone. Your efforts, whether big or small, help us grow and improve. 
Contributions aren’t limited to code—answering questions, helping others, enhancing our 
documentation, and sharing the project are equally impactful.

## Ways to contribute

There are several ways you can contribute to ***OpenR***:

- Fix existing issues in the codebase.
- Report bugs or suggest new features by submitting issues.
- Implement new models or features.
- Improve our examples or contribute to the documentation.

The best way to do that is to open a **Pull Request** and link it to the **issue** that you'd like to work on. We try to give priority to opened PRs as we can easily track the progress of the fix, and if the contributor does not have time anymore, someone else can take the PR over.


## Create a Pull Request

> This guide was heavily inspired by [huggingface guide to contribution](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request)


You will need basic `git` proficiency to contribute to ***OpenR***. While `git` is not the easiest tool to use, it has the greatest
manual. Type `git --help` in a shell and enjoy! If you prefer books, [Pro
Git](https://git-scm.com/book/en/v2) is a very good reference.

Follow the steps below to start contributing:

1. Fork the [repository](https://github.com/openreasoner/openr) by
   clicking on the **[Fork](https://github.com/openreasoner/openr/fork)** button on the repository's page. This creates a copy of the code
   under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   git clone git@github.com:<your Github handle>/openr.git
   cd openr
   git remote add upstream https://github.com/openreasoner/openr.git
   ```

3. Create a new branch to hold your development changes:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   🚨 **Do not** work on the `main` branch!


4. Set up a development environment by following the [README](https://github.com/openreasoner/openr/blob/main/README.md) file.


5. Develop the features in your branch.

   Once you're happy with your changes, add the changed files with `git add` and
   record your changes locally with `git commit`:

   ```bash
   git add modified_file.py
   git commit
   ```

   Please remember to write [good commit
   messages](https://chris.beams.io/posts/git-commit/) to clearly communicate the changes you made!

   To keep your copy of the code up to date with the original
   repository, rebase your branch on `upstream/branch` *before* you open a pull request or if requested by a maintainer:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Push your changes to your branch:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

   If you've already opened a pull request, you'll need to force push with the `--force` flag. Otherwise, if the pull request hasn't been opened yet, you can just push your changes normally.


6. Now you can go to your fork of the repository on GitHub and click on **Pull Request** to open a pull request.
When you're ready, you can send your changes to the project maintainers for review.


7. It's ok if maintainers request changes, it happens to our core contributors
   too! So everyone can see the changes in the pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.

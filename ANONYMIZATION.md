# Anonymization Notes

This code release was staged from the development workspace into a clean
repository directory. The staging process removes local absolute paths and local
usernames from copied source files.

Before pushing, run:

```bash
grep -RInE "<local-user>|<machine-name>|<absolute-local-path>|<private-email>|affiliation" . --exclude-dir=.git
```

The expected result is no author-identifying output. Generic strings such as
anonymous placeholder identities are not author identities.

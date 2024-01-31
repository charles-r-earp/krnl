# publish order
1. krnl-macros 
2. krnl-core
3. krnl
4. krnlc 

# publishing
1. Create a new branch ie "publish-v0.1.0".
2. Bump all crates to the next version, removing the prerelease, ie "=0.1.0".
3. Update krnlc lockfile.
4. Recompile with the new krnlc version. 
5. Set publish to true for workspace / krnlc.
6. Commit and push the new branch.
7. PR to merge with main. Wait for CI and merge.
8. Pull the merged main.
9. Tag main with the version, ie `git tag v0.1.0`.
10. Push the tag `git push origin v0.1.0`.
11. Move into each crate directory and cargo publish.

# bumping next pre-release
1. Set publish to false for workspace / krnlc.
2. Bump all versions to the next version with prerelease "alpha", ie "=0.1.1-alpha".
3. Update krnlc lockfile. 
4. Recompile with the new krnlc version. 
5. Commit and push to main.
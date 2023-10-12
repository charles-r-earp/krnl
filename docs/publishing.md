# publish order
1. krnl-macros 
2. krnl-core
3. krnl
4. krnlc 

# publishing
1. Create a new branch ie "publish-v0.1.0".
2. Bump all crates to the next version, removing the prerelease, ie "=0.1.0".
3. Recompile with the new krnlc version. 
4. Set publish to true for workspace / krnlc.
5. Commit and push the new branch.
6. PR to merge with main. Wait for CI and merge.
7. Pull the merged main.
8. Tag main with the version, ie `git tag v0.1.0`.
9. Push the tag `git push origin v0.1.0`.
10. Move into each crate directory and cargo publish.

# bumping next pre-release
1. Set publish to false for workspace / krnlc.
2. Bump all versions to the next version with prerelease "alpha", ie "=0.1.1-alpha".
3. Commit and push to main.
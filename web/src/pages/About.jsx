import Shell from '../components/Shell';

export default function About() {
  return (
    <Shell active="about">
    <div className="page narrow">
      <div className="kicker">Documentation</div>
      <h1>How it works</h1>
      <ol>
        <li><b>Upload.</b> An IV stack (TIFF or OME-TIFF, planes × height × width) and an EX image or stack of the same region. Give the pixel sizes in µm, the
          channel to match on (GFP / GCaMP), optionally a DAPI channel, and the EX planes to max-project. Your browser reads the files' headers and
          shows previews before anything is uploaded.</li>
        <li><b>Masks.</b> Every IV plane and the EX projection are segmented by three Cellpose models: fine-tuned cyto2 and cyto3 (Cellpose 3) and a
          fine-tuned CP-SAM (Cellpose 4). In the viewer you can show each model, or only the cells two or all three models agree on.</li>
        <li><b>Orientation.</b> PMAC finds how the IV field sits in the EX image (mirror, rotation, scale) from its own appearance descriptors, with no
          prior: every rotation in 10° steps, with and without a mirror, then 1° refinement; the evidence is the number of cell pairs that agree on one
          similarity transform, compared with 50 shuffles (family-wise p). If p ≥ 0.05 the card says the orientation is uncertain. You can accept it,
          use the runner-up, or set it by hand and match again.</li>
        <li><b>Matches.</b> For every IV cell on every plane, PMAC ranks the EX cells. Click an IV cell to see the top candidates, crops, and scores.</li>
        <li><b>Verify.</b> Accept (A), reject (R) or mark unsure (U); 1–9 picks a candidate, clicking any EX cell picks it by hand. Nothing is accepted
          for you: PMAC's scores are suggestions, and its accuracy on data from other labs is not yet known.</li>
        <li><b>Register.</b> From the accepted pairs, a similarity, affine or thin-plate-spline fit per IV plane (pairs within ± Δz) or global, with
          leave-one-out error in µm; blend, swipe and checkerboard overlays. Export the pairs (CSV) and the per-plane transforms (JSON).</li>
      </ol>
      <h2>Privacy and retention</h2>
      <ul>
        <li>Uploads go from your browser to a private Vercel Blob store (US East) through short-lived signed links. Nothing is public; each file
          is read through a signed link that expires within 12 hours.</li>
        <li>The analysis runs on NYU's HPC cluster, which copies your uploads to its scratch storage and then deletes them from the cloud. Results
          (images, masks, matches) stay in the cloud for 7 days after the analysis and are then removed; the copy on NYU HPC scratch remains so the
          results can be restored. Ask the person who gave you the invite to delete a session entirely.</li>
        <li>Anyone who has a session's link can open it and change its verdicts. Share links only with collaborators.</li>
        <li>Do not upload human, clinical or otherwise identifiable data.</li>
      </ul>
      <h2>Limits of the pilot</h2>
      <ul>
        <li>Invite code required; 3 analyses per code per day; up to 900 MB per file; IV up to 400 million voxels.</li>
        <li>Jobs wait for a free GPU on NYU HPC: minutes to hours. The session page shows where your job is.</li>
        <li>Storage and download allowances are small (a free hosting tier), so results expire after 7 days.</li>
      </ul>
    </div>
    </Shell>
  );
}

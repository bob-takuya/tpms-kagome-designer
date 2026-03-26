import { store } from '../store';

export function createStatusBar(container: HTMLElement): void {
  const statusBar = document.createElement('div');
  statusBar.className = 'status-bar';
  statusBar.innerHTML = `
    <span class="status-item">Vertices: <span id="stat-vertices">0</span></span>
    <span class="status-item">Faces: <span id="stat-faces">0</span></span>
    <span class="status-item">Strips: <span id="stat-strips">0</span></span>
    <span class="status-item">Junctions: <span id="stat-junctions">0</span></span>
  `;

  container.appendChild(statusBar);

  // Subscribe to store changes
  store.subscribe((state) => {
    document.getElementById('stat-vertices')!.textContent = String(state.stats.vertices);
    document.getElementById('stat-faces')!.textContent = String(state.stats.faces);
    document.getElementById('stat-strips')!.textContent = String(state.stats.strips);
    document.getElementById('stat-junctions')!.textContent = String(state.stats.junctions);
  });
}

// What React Flow needs from a browser that jsdom lacks (the library's own testing recipe): a ResizeObserver that
// reports at once, a DOMMatrixReadOnly that reads the zoom, element sizes from inline styles, and SVG getBBox.
// Call once at the top of a test file that renders the run graph.

class ResizeObserverStub {
  private readonly callback: ResizeObserverCallback;

  constructor(callback: ResizeObserverCallback) {
    this.callback = callback;
  }

  observe(target: Element) {
    const width = target instanceof HTMLElement ? target.offsetWidth : 0;
    const height = target instanceof HTMLElement ? target.offsetHeight : 0;
    const contentRect = { x: 0, y: 0, top: 0, left: 0, right: width, bottom: height, width, height };
    this.callback([{ target, contentRect } as unknown as ResizeObserverEntry], this as unknown as ResizeObserver);
  }

  unobserve() {}

  disconnect() {}
}

class DOMMatrixReadOnlyStub {
  readonly m22: number;

  constructor(transform?: string) {
    const scale = transform?.match(/scale\(([0-9.]+)\)/)?.[1];
    this.m22 = scale !== undefined ? Number(scale) : 1;
  }
}

export function mockReactFlow(): void {
  window.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver;
  window.DOMMatrixReadOnly = DOMMatrixReadOnlyStub as unknown as typeof DOMMatrixReadOnly;
  Object.defineProperties(window.HTMLElement.prototype, {
    offsetHeight: {
      configurable: true,
      get(this: HTMLElement) {
        return parseFloat(this.style.height) || 1;
      },
    },
    offsetWidth: {
      configurable: true,
      get(this: HTMLElement) {
        return parseFloat(this.style.width) || 1;
      },
    },
  });
  Object.defineProperty(window.SVGElement.prototype, "getBBox", {
    configurable: true,
    value: () => ({ x: 0, y: 0, width: 0, height: 0 }),
  });
}

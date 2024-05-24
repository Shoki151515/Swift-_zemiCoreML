import SwiftUI
import CoreML
import Vision
import AVFoundation

@main
struct ObjectDetectionApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    var body: some View {
        CameraView()
            .edgesIgnoringSafeArea(.all)
    }
}

struct CameraView: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> UIViewController {
        let viewController = CameraViewController()
        return viewController
    }
    
    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {}
}

class CameraViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

    var captureSession: AVCaptureSession!
    var previewLayer: AVCaptureVideoPreviewLayer!
    var detectionOverlay: CALayer! = nil

    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
    }

    func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .photo

        guard let backCamera = AVCaptureDevice.default(for: AVMediaType.video) else {
            print("Unable to access back camera!")
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: backCamera)
            captureSession.addInput(input)

            let output = AVCaptureVideoDataOutput()
            output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
            captureSession.addOutput(output)

            previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
            previewLayer.videoGravity = .resizeAspectFill
            previewLayer.frame = view.frame
            view.layer.addSublayer(previewLayer)

            captureSession.startRunning()
        } catch let error  {
            print("Error Unable to initialize back camera:  \(error.localizedDescription)")
        }

        detectionOverlay = CALayer()
        detectionOverlay.bounds = CGRect(x: 0.0,
                                         y: 0.0,
                                         width: view.bounds.width,
                                         height: view.bounds.height)
        detectionOverlay.position = CGPoint(x: view.bounds.midX, y: view.bounds.midY)
        view.layer.addSublayer(detectionOverlay)
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        var requestOptions: [VNImageOption: Any] = [:]

        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: requestOptions)

        do {
            try imageRequestHandler.perform([self.detectionRequest])
        } catch {
            print(error)
        }
    }

    lazy var detectionRequest: VNCoreMLRequest = {
        do {
            let modelURL = Bundle.main.url(forResource: "best", withExtension: "mlpackage")!
            let model = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))
            return VNCoreMLRequest(model: model, completionHandler: self.handleDetection)
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }()

    func handleDetection(request: VNRequest, error: Error?) {
        guard let observations = request.results as? [VNRecognizedObjectObservation] else {
            return
        }

        DispatchQueue.main.async {
            self.detectionOverlay.sublayers?.removeSubrange(1...)

            for observation in observations {
                let boundingBox = self.transformBoundingBox(observation.boundingBox)
                let boundingBoxPath = CGPath(rect: boundingBox, transform: nil)

                let shapeLayer = CAShapeLayer()
                shapeLayer.path = boundingBoxPath
                shapeLayer.strokeColor = UIColor.red.cgColor
                shapeLayer.fillColor = UIColor.clear.cgColor
                shapeLayer.lineWidth = 2

                self.detectionOverlay.addSublayer(shapeLayer)
            }
        }
    }

    func transformBoundingBox(_ boundingBox: CGRect) -> CGRect {
        let x = boundingBox.origin.x * view.bounds.width
        let y = (1 - boundingBox.origin.y - boundingBox.height) * view.bounds.height
        let width = boundingBox.width * view.bounds.width
        let height = boundingBox.height * view.bounds.height
        return CGRect(x: x, y: y, width: width, height: height)
    }
}

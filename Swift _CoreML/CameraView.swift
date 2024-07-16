import SwiftUI


// UIViewControllerをSwiftUIに統合するためのプロトコルを実装
struct CameraView: UIViewControllerRepresentable {
    // UIViewControllerを作成する
    func makeUIViewController(context: Context) -> UIViewController {
        let viewController = CameraViewController()
        return viewController
    }
    
    
    // UIViewControllerの更新方法を定義（今回は何もしない）
    func updateUIViewController(_ uiViewController: UIViewController, context: Context) {}
}

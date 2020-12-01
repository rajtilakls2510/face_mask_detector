package com.example.facemaskdetector;

import android.annotation.SuppressLint;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.media.Image;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity {


    private static final String TAG = "MainActivity";
    private int REQUEST_CODE_PERMISSIONS = 101;
    private final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA"};
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private PreviewView previewView;
    FaceDetector detector;
    ImageView ivBitmap;
    Classifier classifier;
    Button increaseFaces;
    private Handler handler;
    private HandlerThread handlerThread;
    private boolean isProcessingFrame=false;

    int maxNumFaces = 3;

    private BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {

        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS) {
                Log.i(TAG, "OpenCV loaded successfully");
            } else {
                super.onManagerConnected(status);
            }
        }
    };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        previewView = findViewById(R.id.previewView);
        ivBitmap = findViewById(R.id.ivBitmap);
        increaseFaces = findViewById(R.id.increase_face);
        increaseFaces.setText("Detecting max "+maxNumFaces+ " faces. Click to increase");

        FaceDetectorOptions options =
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                        .build();
        detector = FaceDetection.getClient(options);

        if (allPermissionsGranted()) {
            startCamera();

        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
        increaseFaces.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                maxNumFaces = (maxNumFaces+1)%5;
                if(maxNumFaces ==0){
                    increaseFaces.setText("Detecting All faces. Click to decrease");
                    Toast.makeText(MainActivity.this, "Warning! Lag will increase", Toast.LENGTH_LONG).show();
                }
                else
                    increaseFaces.setText("Detecting max "+maxNumFaces+ " face(s). Click to increase");
            }
        });
    }





    private void startCamera() {

        cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                // No errors need to be handled for this Future.
                // This should never be reached.
            }
        }, ContextCompat.getMainExecutor(this));

    }

    void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder()
                .build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        ImageAnalysis imageAnalysis =
                new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(1080, 720))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this), new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy imageProxy) {
                int rotationDegrees = imageProxy.getImageInfo().getRotationDegrees();

                @SuppressLint("UnsafeExperimentalUsageError") Image mediaImage = imageProxy.getImage();
                if (mediaImage != null) {
                    InputImage image =
                            InputImage.fromMediaImage(mediaImage, rotationDegrees);
                    Task<List<Face>> result =
                            detector.process(image)
                                    .addOnSuccessListener(
                                            new OnSuccessListener<List<Face>>() {
                                                @Override
                                                public void onSuccess(List<Face> faces) {
                                                        Mat imageGrab = imageToMat(mediaImage);
                                                        if(!isProcessingFrame)
                                                            processFaces(imageGrab, faces,mediaImage.getHeight(), mediaImage.getWidth());
                                                    }
                                            })
                                    .addOnFailureListener(
                                            new OnFailureListener() {
                                                @Override
                                                public void onFailure(@NonNull Exception e) {
                                                    Log.d(TAG, "onFailure: Failed to Detect");
                                                }
                                            })
                                    .addOnCompleteListener(
                                            new OnCompleteListener<List<Face>>() {
                                                @Override
                                                public void onComplete(@NonNull Task<List<Face>> task) {
                                                    imageProxy.close();
                                                }
                                            });
                }
               
            }
        });

        Camera camera = cameraProvider.bindToLifecycle((LifecycleOwner)this, cameraSelector, preview, imageAnalysis);

    }

     public synchronized void runInBackground(Runnable r)
    {
        if(handler!=null)
            handler.post(r);
    }

     public void processFaces(Mat imageGrab, List<Face> faces, int height, int width)
    {
        isProcessingFrame=true;
        runInBackground(new Runnable() {
            @Override
            public void run() {

                Mat bgrMat = new Mat(height,width,CvType.CV_8UC4);

                Imgproc.cvtColor(imageGrab, bgrMat, Imgproc.COLOR_YUV2BGR_I420);
                Core.rotate( bgrMat, bgrMat, Core.ROTATE_90_CLOCKWISE);
                ArrayList<Face> newfaces = getMaxFaces (faces, maxNumFaces);
                long startTime = SystemClock.uptimeMillis();
                for (Face face : newfaces) {
                    Rect bounds = face.getBoundingBox();
                    float rotY = face.getHeadEulerAngleY();  // Head is rotated to the right rotY degrees
                    float rotZ = face.getHeadEulerAngleZ();  // Head is tilted sideways rotZ degrees
                    try{

                        Mat croppedFace =new Mat(bgrMat,new org.opencv.core.Rect( bounds.left,bounds.top, bounds.right - bounds.left, bounds.bottom-bounds.top ));

                        Mat resizedimage = new Mat();
                        org.opencv.core.Size sz = new org.opencv.core.Size(224,224);
                        Imgproc.resize( croppedFace, resizedimage, sz );
                        Imgproc.cvtColor(resizedimage, resizedimage, Imgproc.COLOR_BGR2RGB, 3);


                        int prediction = classifier.classifyMat(resizedimage);
                        if (prediction == 0) {
                            if(maxNumFaces!=0)
                                Imgproc.putText(bgrMat, "With Mask", new Point(bounds.left, bounds.top), Core.FONT_HERSHEY_COMPLEX, 0.8, new Scalar(0, 255, 0), 1);

                            Imgproc.rectangle(bgrMat,
                                    new Point(bounds.left, bounds.top),
                                    new Point(bounds.right, bounds.bottom),
                                    new Scalar(0, 255, 0), 2);
                        }
                        else
                        {
                            if (maxNumFaces!=0)
                                Imgproc.putText(bgrMat,"No Mask",new Point(bounds.left,bounds.top), Core.FONT_HERSHEY_COMPLEX, 0.5, new Scalar(0,0,255),1 );
                            Imgproc.rectangle(bgrMat,
                                    new Point(bounds.left,bounds.top),
                                    new Point(bounds.right, bounds.bottom),
                                    new Scalar(0,0,255),2);
                        }
                        Mat rgbaMatOut = new Mat();
                        Imgproc.cvtColor(bgrMat, rgbaMatOut, Imgproc.COLOR_BGR2RGBA, 0);

                        final Bitmap bitmap = Bitmap.createBitmap(bgrMat.cols(), bgrMat.rows(), Bitmap.Config.ARGB_8888);
                        Utils.matToBitmap(rgbaMatOut, bitmap);
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                ivBitmap.setImageBitmap(bitmap);
                            }
                        });

                    }catch (Exception e){

                    }

                }
                long endTime = SystemClock.uptimeMillis();
                Mat rgbaMatOut = new Mat();
                Imgproc.cvtColor(bgrMat, rgbaMatOut, Imgproc.COLOR_BGR2RGBA, 0);

                final Bitmap bitmap = Bitmap.createBitmap(bgrMat.cols(), bgrMat.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(rgbaMatOut, bitmap);
                Log.d(TAG, "onSuccess: Prediction time: "+ (endTime-startTime));
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        ivBitmap.setImageBitmap(bitmap);
                        isProcessingFrame=false;
                    }
                });

            }
        });


    }

    private int getMaxAreaFace(ArrayList<Integer> areas)

    {
        Integer maxArea=0;
        int maxAreaIndex=0;
        for(int i=0;i<areas.size();i++) {
            if(areas.get(i)>maxArea)
            {
                maxArea=areas.get(i);
                maxAreaIndex=i;
            }
        }
        return maxAreaIndex;

    }
    private ArrayList<Face> getMaxFaces(List<Face> faces, int numFaces ) {
        ArrayList<Face> newFaces = new ArrayList<>();
        ArrayList<Integer> areas = new ArrayList<>();
        ArrayList<Face> oldFaces =new ArrayList<>();



        for(Face face : faces){
            oldFaces.add(face);
            Rect bounds = face.getBoundingBox();
            areas.add((bounds.right- bounds.left)*(bounds.bottom-bounds.top));
        }

        int counter=0;
        if(numFaces!=0) {
            while (counter < numFaces && !oldFaces.isEmpty()) {
                int index = getMaxAreaFace(areas);
                newFaces.add(oldFaces.get(index));
                areas.remove(index);

                oldFaces.remove(index);
                counter++;
            }


            return newFaces;
        }
        return oldFaces;

    }


    public static Mat imageToMat(Image image) {
        ByteBuffer buffer;
        int rowStride;
        int pixelStride;
        int width = image.getWidth();
        int height = image.getHeight();
        int offset = 0;

        Image.Plane[] planes = image.getPlanes();
        byte[] data = new byte[image.getWidth() * image.getHeight() * ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8];
        byte[] rowData = new byte[planes[0].getRowStride()];

        for (int i = 0; i < planes.length; i++) {
            buffer = planes[i].getBuffer();
            rowStride = planes[i].getRowStride();
            pixelStride = planes[i].getPixelStride();
            int w = (i == 0) ? width : width / 2;
            int h = (i == 0) ? height : height / 2;
            for (int row = 0; row < h; row++) {
                int bytesPerPixel = ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8;
                if (pixelStride == bytesPerPixel) {
                    int length = w * bytesPerPixel;
                    buffer.get(data, offset, length);

                    if (h - row != 1) {
                        buffer.position(buffer.position() + rowStride - length);
                    }
                    offset += length;
                } else {


                    if (h - row == 1) {
                        buffer.get(rowData, 0, width - pixelStride + 1);
                    } else {
                        buffer.get(rowData, 0, rowStride);
                    }

                    for (int col = 0; col < w; col++) {
                        data[offset++] = rowData[col * pixelStride];
                    }
                }
            }
        }

        Mat mat = new Mat(height + height / 2, width, CvType.CV_8UC1);
        mat.put(0, 0, data);

        return mat;
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    private boolean allPermissionsGranted() {

        for (String permission: REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    protected synchronized void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug())
        {
            Log.d(TAG, "onResume: Opencv Loaded Successfully");
        }
        else
        {
            Log.d(TAG, "onResume: Opencv Loading Unsuccessful");
        }
        try {
            classifier = new Classifier(MainActivity.this);
        } catch (IOException e) {
            Log.d(TAG, "onResume: Failed to load Classifier." + e.getMessage());
        }
        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
    }

    @Override
    protected synchronized void onPause() {

        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException e) {
            Log.d(TAG, "onPause: HandlerThread Quit Exception");
        }
        super.onPause();
    }
}

